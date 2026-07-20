"""Cross-provider failover routing.

Exercises the classified candidate walk in Solwyn/AsyncSolwyn._intercepted_call
with faked provider clients. Clients are MagicMocks whose
``__class__.__module__`` triggers adapter detection and whose
``with_options(...)`` returns the same mock (so the per-hop bound resolves back
to the configured client).

No real provider SDKs are importable, so transport exceptions are duck-typed:
a ``_Status(status_code=429)`` classifies as FAILOVER (advance the chain) while
``_Status(status_code=400)`` classifies as FAIL_FAST (stop the chain).
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._types import CircuitState
from solwyn.client import AsyncSolwyn, Deadline, Solwyn
from solwyn.config import SolwynConfig
from solwyn.exceptions import ProviderUnavailableError, UntranslatableRequestError


class _Status(Exception):
    """Duck-typed transport error carrying an HTTP ``status_code``.

    classify_exception reads ``status_code``: 429/529 -> FAILOVER, 4xx ->
    FAIL_FAST, 5xx -> POST_SEND_AMBIGUOUS.
    """

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


# ── client fakes ─────────────────────────────────────────────────────────


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    return client


def _anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"
    client.with_options.return_value = client
    return client


class _GoogleClient:
    def __init__(self) -> None:
        self.models = SimpleNamespace(
            generate_content=MagicMock(return_value=_google_response()),
            generate_content_stream=MagicMock(return_value=iter([_google_chunk()])),
        )


_GoogleClient.__module__ = "google.genai.client"


def _google_client() -> _GoogleClient:
    return _GoogleClient()


def _openai_response() -> SimpleNamespace:
    # Native OpenAI Chat Completions shape (duck-typed): the adapter reads
    # ``usage`` and the normalizer reads ``choices[0].message`` + ``model``.
    message = SimpleNamespace(role="assistant", content="ok from gpt", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_response() -> SimpleNamespace:
    # Native Anthropic Messages shape (duck-typed). When this response crosses
    # back to an OpenAI-dialect caller it is reshaped by normalize_response, so
    # it must expose the real ``content``/``stop_reason``/``model`` access paths
    # (a bare MagicMock would fail CanonicalResponse validation on ``model``).
    block = SimpleNamespace(type="text", text="ok from claude")
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _google_response() -> SimpleNamespace:
    return SimpleNamespace(
        model="gemini-3.5-flash",
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=0,
        ),
    )


def _google_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=0,
        )
    )


def _allow_budget(*, failover_tuning_allowed: bool | None = None) -> SimpleNamespace:
    """Allow result with no reservation (skips confirm_cost)."""
    return SimpleNamespace(
        allowed=True,
        reservation_id=None,
        project_id=VALID_PROJECT_ID,
        price_hints=None,
        failover_tuning_allowed=failover_tuning_allowed,
    )


def _make_solwyn(client: object, **overrides: object) -> Solwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    solwyn._reporter.report = MagicMock()
    return solwyn


def _close(solwyn: Solwyn) -> None:
    solwyn._reporter._http.close()
    solwyn._budget._http.close()


_PLAIN_REQUEST = {
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "hi"}],
}

_FAILOVER_TUNING_FIELDS = (
    "failover_total_timeout",
    "failover_idempotency",
    "same_provider_retries",
    "circuit_breaker_recovery_timeout_jitter",
    "circuit_breaker_failure_threshold",
    "circuit_breaker_recovery_timeout",
    "circuit_breaker_success_threshold",
)
_CUSTOM_FAILOVER_TUNING = {
    "failover_total_timeout": 91.0,
    "failover_idempotency": "never",
    "same_provider_retries": 4,
    "circuit_breaker_recovery_timeout_jitter": 0.05,
    "circuit_breaker_failure_threshold": 8,
    "circuit_breaker_recovery_timeout": 75,
    "circuit_breaker_success_threshold": 6,
}


def _current_failover_tuning(solwyn: Solwyn | AsyncSolwyn) -> dict[str, object]:
    return {name: getattr(solwyn._config, name) for name in _FAILOVER_TUNING_FIELDS}


# ── cross-provider failover (the core case) ──────────────────────────────


@pytest.mark.unit
class TestCrossProviderFailover:
    def test_openai_down_anthropic_served(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create.return_value = anthropic_resp

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Anthropic served the request. The plain-text request crossed via real
        # translation, and the served response was normalized back into the
        # OpenAI dialect the caller wrote — so the native access path resolves
        # to the Anthropic text (NOT identity with the raw Anthropic object).
        assert result is not anthropic_resp
        assert result.choices[0].message.content == "ok from claude"
        anthropic.messages.create.assert_called_once()
        openai.chat.completions.create.assert_called_once()
        # The Anthropic call received TRANSLATED, Anthropic-native kwargs.
        anthropic_kwargs = anthropic.messages.create.call_args.kwargs
        assert anthropic_kwargs["max_tokens"] == 256  # from entry default_params
        assert anthropic_kwargs["model"] == "claude-sonnet-5"  # fallback entry model
        # Translation reshaped the OpenAI message into Anthropic block form.
        assert anthropic_kwargs["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        # The OpenAI-only key never crosses to the Anthropic call.
        assert "max_completion_tokens" not in anthropic_kwargs

        _close(solwyn)

    def test_breaker_accounting_one_failure_one_success(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")
        anthropic_cb = solwyn._get_circuit_breaker("anthropic")
        # Spy record_success so the success assertion is not vacuous (a never-
        # touched breaker would also read failure_count == 0 / CLOSED).
        with (
            patch.object(
                anthropic_cb, "record_success", wraps=anthropic_cb.record_success
            ) as served_success,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Exactly one failure recorded on openai, one success on anthropic.
        assert openai_cb.failure_count == 1
        served_success.assert_called_once()
        assert anthropic_cb.state == CircuitState.CLOSED
        assert anthropic_cb.failure_count == 0

        _close(solwyn)

    def test_cross_provider_success_event_attribution(self) -> None:
        # The success MetadataEvent must be attributed to the SERVED provider,
        # flagged as a provider fallback, and carry requested_provider/model +
        # failover_reason for the dashboard. Here the primary
        # is CLOSED, ATTEMPTED, and 429s — a REACTIVE failover -> PRIMARY_ERROR.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        ev = success[0]
        assert ev.provider.value == "anthropic"  # served, not requested
        assert ev.model == "claude-sonnet-5"
        assert ev.is_provider_fallback is True
        assert ev.is_model_fallback is False
        assert ev.requested_provider.value == "openai"
        assert ev.requested_model == "gpt-5.5"
        # Primary was attempted-and-errored -> reactive failover -> PRIMARY_ERROR.
        assert ev.failover_reason is not None and ev.failover_reason.value == "primary_error"
        assert ev.attempt_index == 1

        _close(solwyn)

    def test_primary_errored_success_reason_is_primary_error(self) -> None:
        # PRIMARY CLOSED, ATTEMPTED in this walk, raises 429 ->
        # the cross-provider success event's failover_reason is PRIMARY_ERROR
        # (reactive failover), NOT CIRCUIT_OPEN.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        # Primary breaker starts CLOSED, so the primary IS attempted in the walk.
        assert solwyn._get_circuit_breaker("openai").state == CircuitState.CLOSED
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Primary WAS dispatched (and errored); fallback served.
        openai.chat.completions.create.assert_called_once()
        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        assert success[0].is_provider_fallback is True
        assert success[0].failover_reason is not None
        assert success[0].failover_reason.value == "primary_error"

        _close(solwyn)

    def test_primary_open_not_eligible_skips_straight_to_fallback(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)
        # Force the primary breaker OPEN and not recovery-eligible.
        openai_cb = solwyn._get_circuit_breaker("openai")
        for _ in range(3):
            openai_cb.record_failure()
        assert openai_cb.state == CircuitState.OPEN
        assert openai_cb.recovery_eligible is False

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Primary was NOT attempted; fallback served directly.
        openai.chat.completions.create.assert_not_called()
        anthropic.messages.create.assert_called_once()
        assert result is not None

        # the primary was SKIPPED (breaker pre-OPEN), never
        # attempted in this walk -> proactive reroute -> CIRCUIT_OPEN.
        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        assert success[0].is_provider_fallback is True
        assert success[0].failover_reason is not None
        assert success[0].failover_reason.value == "circuit_open"
        # Fix [B]: attempt_index is the served runtime's position in the
        # CONFIGURED chain (1 == first fallback), NOT the candidate-walk index.
        # Even though the OPEN-not-eligible primary was dropped from the
        # health-filtered candidate list (so the fallback was attempted at
        # walk-index 0), the chain-depth funnel must read attempt_index == 1.
        assert success[0].attempt_index == 1

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_primary_open_attempt_index_is_chain_position(self) -> None:
        # Async mirror of fix [B]: a pre-OPENed primary drops out of the
        # candidate list, but the served first-fallback's success event still
        # carries its CONFIGURED-chain attempt_index == 1.
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(return_value=_openai_response())
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)
        openai_cb = solwyn._get_circuit_breaker("openai")
        for _ in range(3):
            openai_cb.record_failure()
        assert openai_cb.state == CircuitState.OPEN
        assert openai_cb.recovery_eligible is False

        with patch.object(
            solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        openai.chat.completions.create.assert_not_awaited()
        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        assert success[0].is_provider_fallback is True
        assert success[0].attempt_index == 1
        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_openai_down_anthropic_served(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status(429))
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create = AsyncMock(return_value=anthropic_resp)

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        solwyn._reporter.report = MagicMock()
        openai_cb = solwyn._get_circuit_breaker("openai")
        anthropic_cb = solwyn._get_circuit_breaker("anthropic")

        with patch.object(
            solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Response normalized back to the OpenAI dialect (not the raw Anthropic
        # object); the native access path yields the Anthropic text.
        assert result is not anthropic_resp
        assert result.choices[0].message.content == "ok from claude"
        anthropic.messages.create.assert_awaited_once()
        anthropic_kwargs = anthropic.messages.create.call_args.kwargs
        assert anthropic_kwargs["max_tokens"] == 256
        assert anthropic_kwargs["model"] == "claude-sonnet-5"
        # Breaker accounting mirrors the sync case.
        assert openai_cb.failure_count == 1
        assert anthropic_cb.failure_count == 0

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ── same-provider model swap ─────────────────────────────────────────────


@pytest.mark.unit
class TestSameProviderModelSwap:
    def test_model_swap_on_same_client(self) -> None:
        # Primary model fails (429); the SAME client serves the gpt-5.4-mini swap.
        client = _openai_client()
        success = _openai_response()
        client.chat.completions.create.side_effect = [_Status(429), success]

        solwyn = _make_solwyn(client, model="gpt-5.5", fallback=[(client, "gpt-5.4-mini")])
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-5.4-mini"

        success_events = [e for e in events if e.status.value == "success"]
        assert len(success_events) == 1
        assert success_events[0].is_model_fallback is True
        assert success_events[0].is_provider_fallback is False

        _close(solwyn)


# ── fail-fast (4xx stops the chain) ──────────────────────────────────────


@pytest.mark.unit
class TestFailFast:
    def test_400_stops_chain_and_reraises_original(self) -> None:
        openai = _openai_client()
        original = _Status(400, "bad request")
        openai.chat.completions.create.side_effect = original
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # A 400 classifies FAIL_FAST: the chain STOPS, the fallback is NOT
        # attempted, and the ORIGINAL exception propagates unchanged.
        assert exc_info.value is original
        anthropic.messages.create.assert_not_called()
        openai.chat.completions.create.assert_called_once()

        # Disposition table: FAIL_FAST is request-shaped, NOT a provider-
        # health signal, so it must NOT count the breaker.
        assert openai_cb.failure_count == 0
        assert openai_cb.state == CircuitState.CLOSED

        _close(solwyn)

    def test_post_send_ambiguous_counts_breaker_but_does_not_failover(self) -> None:
        # A 500 classifies POST_SEND_AMBIGUOUS: it IS a health signal (counts the
        # breaker) but the request may have run, so the chain must NOT fail over
        # under the default "safe" idempotency — it re-raises the original.
        openai = _openai_client()
        original = _Status(500, "server error")
        openai.chat.completions.create.side_effect = original
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value is original
        anthropic.messages.create.assert_not_called()  # post-send: never failover
        assert openai_cb.failure_count == 1  # but it DOES count the breaker

        _close(solwyn)


# ── chain exhaustion ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestChainExhaustion:
    def test_all_open_raises_provider_unavailable(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        # Open BOTH breakers, not recovery-eligible.
        for name in ("openai", "anthropic"):
            cb = solwyn._get_circuit_breaker(name)
            for _ in range(3):
                cb.record_failure()
            assert cb.recovery_eligible is False

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(ProviderUnavailableError) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Neither client was dispatched.
        openai.chat.completions.create.assert_not_called()
        anthropic.messages.create.assert_not_called()
        # The error carries the attempted chain (empty here — none were eligible).
        assert hasattr(exc_info.value, "attempted")

        _close(solwyn)

    def test_all_attempted_providers_fail_reraises_last_exception(self) -> None:
        openai = _openai_client()
        first = _Status(429, "primary failed")
        openai.chat.completions.create.side_effect = first
        anthropic = _anthropic_client()
        last = _Status(429, "fallback failed")
        anthropic.messages.create.side_effect = last

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value is last
        assert solwyn._get_circuit_breaker("openai").failure_count == 1
        assert solwyn._get_circuit_breaker("anthropic").failure_count == 1

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_all_attempted_providers_fail_reraises_last_exception(self) -> None:
        openai = _openai_client()
        first = _Status(429, "primary failed")
        openai.chat.completions.create = AsyncMock(side_effect=first)
        anthropic = _anthropic_client()
        last = _Status(429, "fallback failed")
        anthropic.messages.create = AsyncMock(side_effect=last)

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
            pytest.raises(_Status) as exc_info,
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value is last
        assert solwyn._get_circuit_breaker("openai").failure_count == 1
        assert solwyn._get_circuit_breaker("anthropic").failure_count == 1

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()


# ── server failover-tuning directive ────────────────────────────────────


@pytest.mark.unit
class TestFailoverTuningDirective:
    def test_false_then_true_suppresses_and_restores_sync_tuning_once(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            budget_check_cache_ttl=17,
            **_CUSTOM_FAILOVER_TUNING,
        )
        runtime_order = [runtime.entry for runtime in solwyn._runtimes]
        breaker_ids = {
            provider: id(breaker) for provider, breaker in solwyn._circuit_breakers.items()
        }
        primary_breaker = solwyn._circuit_breakers["openai"]
        caplog.set_level("WARNING", logger="solwyn._base")

        with (
            patch.object(
                solwyn._budget,
                "check_budget",
                side_effect=[
                    _allow_budget(failover_tuning_allowed=False),
                    _allow_budget(failover_tuning_allowed=True),
                    _allow_budget(failover_tuning_allowed=False),
                ],
            ),
            patch.object(
                primary_breaker,
                "replace_tuning",
                wraps=primary_breaker.replace_tuning,
            ) as replace_tuning,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)
            defaults = {
                name: SolwynConfig.model_fields[name].default for name in _FAILOVER_TUNING_FIELDS
            }
            assert _current_failover_tuning(solwyn) == defaults
            assert openai.with_options.call_args.kwargs["timeout"] <= 30.0

            solwyn.chat.completions.create(**_PLAIN_REQUEST)
            assert _current_failover_tuning(solwyn) == _CUSTOM_FAILOVER_TUNING

            solwyn.chat.completions.create(**_PLAIN_REQUEST)
            assert replace_tuning.call_count == 3
            solwyn._apply_failover_tuning_directive(False)
            assert replace_tuning.call_count == 3

        assert openai.chat.completions.create.call_count == 3
        assert [runtime.entry for runtime in solwyn._runtimes] == runtime_order
        assert {
            provider: id(breaker) for provider, breaker in solwyn._circuit_breakers.items()
        } == breaker_ids
        assert solwyn._config.budget_check_cache_ttl == 17
        suppression_logs = [
            record
            for record in caplog.records
            if record.name == "solwyn._base" and "failover tuning" in record.getMessage().lower()
        ]
        assert len(suppression_logs) == 1
        assert suppression_logs[0].getMessage() == (
            "Custom failover tuning is unavailable for this plan; SDK defaults applied"
        )

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_false_suppresses_tuning_but_async_provider_call_still_runs(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(return_value=_openai_response())
        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            **_CUSTOM_FAILOVER_TUNING,
        )
        runtime_order = [runtime.entry for runtime in solwyn._runtimes]
        solwyn._reporter.report = MagicMock()

        with patch.object(
            solwyn._budget,
            "check_budget",
            new=AsyncMock(return_value=_allow_budget(failover_tuning_allowed=False)),
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        defaults = {
            name: SolwynConfig.model_fields[name].default for name in _FAILOVER_TUNING_FIELDS
        }
        assert _current_failover_tuning(solwyn) == defaults
        assert [runtime.entry for runtime in solwyn._runtimes] == runtime_order
        openai.chat.completions.create.assert_awaited_once()
        assert openai.with_options.call_args.kwargs["timeout"] <= 30.0

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    def test_missing_or_fail_open_directive_is_a_noop_and_call_proceeds(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            **_CUSTOM_FAILOVER_TUNING,
        )

        with patch.object(
            solwyn._budget,
            "check_budget",
            return_value=_allow_budget(failover_tuning_allowed=None),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert _current_failover_tuning(solwyn) == _CUSTOM_FAILOVER_TUNING
        openai.chat.completions.create.assert_called_once()
        _close(solwyn)

    def test_deadline_can_replace_total_without_resetting_start(self) -> None:
        deadline = Deadline(30.0)
        original_start = deadline._start

        deadline.replace_total(10.0)

        assert deadline._start == original_start
        assert 0.0 < deadline.remaining() <= 10.0


# ── per-hop deadline bound ───────────────────────────────────────────────


@pytest.mark.unit
class TestPerHopDeadline:
    def test_with_options_called_with_max_retries_zero_and_finite_timeout(self) -> None:
        client = _openai_client()
        client.chat.completions.create.return_value = _openai_response()

        solwyn = _make_solwyn(client, model="gpt-5.5")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        client.with_options.assert_called_once()
        call = client.with_options.call_args
        assert call.kwargs["max_retries"] == 0
        timeout = call.kwargs["timeout"]
        assert isinstance(timeout, int | float)
        assert 0.0 < timeout < float("inf")

        _close(solwyn)

    def test_google_dispatch_sets_per_request_timeout_and_retry_bound(self) -> None:
        client = _google_client()
        solwyn = _make_solwyn(client, model="gemini-3.5-flash")

        solwyn._sync_dispatch(
            solwyn._runtimes[0],
            {
                "model": "gemini-3.5-flash",
                "contents": "hi",
                "config": {
                    "temperature": 0.2,
                    "http_options": {"headers": {"x-test": "1"}, "timeout": 999_999},
                },
            },
            is_streaming=False,
            timeout=0.25,
            max_retries=0,
        )

        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["config"]["temperature"] == 0.2
        assert kwargs["config"]["http_options"]["headers"] == {"x-test": "1"}
        assert kwargs["config"]["http_options"]["timeout"] == 250
        assert kwargs["config"]["http_options"]["retry_options"]["attempts"] == 1

        _close(solwyn)

    def test_slow_google_dispatch_raises_within_per_hop_timeout(self) -> None:
        client = _google_client()

        def slow_generate_content(**kwargs: object) -> SimpleNamespace:
            config = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
            http_options = config.get("http_options", {}) if isinstance(config, dict) else {}
            timeout_ms = http_options.get("timeout") if isinstance(http_options, dict) else None
            if isinstance(timeout_ms, int) and timeout_ms < 50:
                raise _Status(429, "google timed out")
            time.sleep(0.05)
            return _google_response()

        client.models.generate_content = slow_generate_content
        solwyn = _make_solwyn(client, model="gemini-3.5-flash")

        started = time.perf_counter()
        with pytest.raises(_Status):
            solwyn._sync_dispatch(
                solwyn._runtimes[0],
                {"model": "gemini-3.5-flash", "contents": "hi"},
                is_streaming=False,
                timeout=0.001,
                max_retries=0,
            )
        elapsed = time.perf_counter() - started

        assert elapsed < 0.03
        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_google_dispatch_sets_per_request_timeout_and_retry_bound(self) -> None:
        client = _google_client()
        client.models.generate_content = AsyncMock(return_value=_google_response())
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, model="gemini-3.5-flash")

        await solwyn._async_dispatch(
            solwyn._runtimes[0],
            {
                "model": "gemini-3.5-flash",
                "contents": "hi",
                "config": {"http_options": {"timeout": 999_999}},
            },
            is_streaming=False,
            timeout=0.125,
            max_retries=0,
        )

        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["config"]["http_options"]["timeout"] == 125
        assert kwargs["config"]["http_options"]["retry_options"]["attempts"] == 1

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_slow_google_dispatch_raises_within_per_hop_timeout(self) -> None:
        client = _google_client()

        async def slow_generate_content(**kwargs: object) -> SimpleNamespace:
            config = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
            http_options = config.get("http_options", {}) if isinstance(config, dict) else {}
            timeout_ms = http_options.get("timeout") if isinstance(http_options, dict) else None
            if isinstance(timeout_ms, int) and timeout_ms < 50:
                raise _Status(429, "google timed out")
            await asyncio.sleep(0.05)
            return _google_response()

        client.models.generate_content = slow_generate_content
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, model="gemini-3.5-flash")

        started = time.perf_counter()
        with pytest.raises(_Status):
            await solwyn._async_dispatch(
                solwyn._runtimes[0],
                {"model": "gemini-3.5-flash", "contents": "hi"},
                is_streaming=False,
                timeout=0.001,
                max_retries=0,
            )
        elapsed = time.perf_counter() - started

        assert elapsed < 0.03
        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    def test_per_hop_timeout_shrinks_across_candidates(self) -> None:
        # Per-hop timeout = remaining / (candidates not yet attempted).
        # With a 30s chain budget and 2 candidates, hop0 gets ~total/2 (NOT the
        # full remaining); the final hop gets the whole remaining slice.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            failover_total_timeout=30.0,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        hop0 = openai.with_options.call_args.kwargs["timeout"]
        hop1 = anthropic.with_options.call_args.kwargs["timeout"]
        # hop0 was divided by 2 candidates (≈15), proving it is not the full 30s.
        assert hop0 < 25.0
        # The last hop gets the remaining slice (÷1), strictly larger than hop0.
        assert hop1 > hop0
        assert hop1 <= 30.0

        _close(solwyn)

    def test_deadline_expired_stops_chain_and_reports_attempted(self) -> None:
        # A zero chain budget means the deadline is already spent by the time the
        # walk begins: NO candidate is dispatched and ProviderUnavailableError
        # carries the full attempted chain.
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            failover_total_timeout=0.0,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(ProviderUnavailableError) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        openai.chat.completions.create.assert_not_called()
        anthropic.messages.create.assert_not_called()
        assert exc_info.value.attempted == ["openai", "anthropic"]

        _close(solwyn)

    def test_deadline_expires_between_hops(self) -> None:
        # Primary fails over (429) but the deadline expires before the second hop:
        # only the primary is attempted and the original failure surfaces.
        openai = _openai_client()
        primary_err = _Status(429)
        openai.chat.completions.create.side_effect = primary_err
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        class _FakeDeadline:
            def __init__(self, total: float) -> None:
                self._expiries = [False, True]  # attempt hop0, then expire

            def expired(self) -> bool:
                return self._expiries.pop(0) if self._expiries else True

            def remaining(self) -> float:
                return 5.0

        with (
            patch("solwyn.client.Deadline", _FakeDeadline),
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value is primary_err
        openai.chat.completions.create.assert_called_once()
        anthropic.messages.create.assert_not_called()

        _close(solwyn)


# ── HALF_OPEN recovery through dispatch ──────────────────────────────────


@pytest.mark.unit
class TestHalfOpenRecoveryThroughDispatch:
    def test_open_eligible_probe_success_closes_breaker(self) -> None:
        client = _openai_client()
        client.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(
            client,
            model="gpt-5.5",
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )

        cb = solwyn._get_circuit_breaker("openai")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.recovery_eligible is True

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert cb.state == CircuitState.CLOSED
        assert cb.success_count == 0
        assert cb.failure_count == 0

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_open_eligible_probe_success_closes_breaker(self) -> None:
        client = _openai_client()
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        solwyn = AsyncSolwyn(
            client,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )

        cb = solwyn._get_circuit_breaker("openai")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.recovery_eligible is True

        with patch.object(
            solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert cb.state == CircuitState.CLOSED
        assert cb.success_count == 0
        assert cb.failure_count == 0

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()


# ── probe-slot release on no-health-signal dispatch exits ────────────────


@pytest.mark.unit
class TestProbeSlotReleaseOnNoHealthSignalExit:
    """A HALF_OPEN probe slot consumed at admit() must be freed when the
    hop exits WITHOUT a health verdict — a cross-provider translation abort or a
    FAIL_FAST raised after the slot was taken. Otherwise the only probe slot is
    permanently occupied and the provider is stranded HALF_OPEN forever.
    """

    def test_fail_fast_in_half_open_frees_probe_slot(self) -> None:
        client = _openai_client()
        client.chat.completions.create.side_effect = _Status(400, "bad request")
        solwyn = _make_solwyn(
            client,
            model="gpt-5.5",
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )
        cb = solwyn._get_circuit_breaker("openai")
        cb.record_failure()  # -> OPEN, recovery-eligible (recovery_timeout=0)
        assert cb.state == CircuitState.OPEN

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # FAIL_FAST is request-shaped, not a health verdict: state stays HALF_OPEN,
        # but the consumed probe slot must be released for the next caller.
        assert cb.state == CircuitState.HALF_OPEN
        assert cb._half_open_probe_active is False
        admission = cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is True

        _close(solwyn)

    def test_translation_abort_in_half_open_frees_probe_slot(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)  # FAILOVER off primary
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )
        anthropic_cb = solwyn._get_circuit_breaker("anthropic")
        anthropic_cb.record_failure()  # -> OPEN, recovery-eligible
        assert anthropic_cb.state == CircuitState.OPEN

        # n>1 is outside the subset -> the cross-provider hop's translation
        # RAISES UntranslatableRequestError before any anthropic network call.
        request = {**_PLAIN_REQUEST, "n": 2}
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableRequestError),
        ):
            solwyn.chat.completions.create(**request)

        anthropic.messages.create.assert_not_called()  # aborted before dispatch
        assert anthropic_cb.state == CircuitState.HALF_OPEN
        assert anthropic_cb._half_open_probe_active is False
        admission = anthropic_cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is True

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_fail_fast_in_half_open_frees_probe_slot(self) -> None:
        client = _openai_client()
        client.chat.completions.create = AsyncMock(side_effect=_Status(400, "bad request"))
        solwyn = AsyncSolwyn(
            client,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )
        cb = solwyn._get_circuit_breaker("openai")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with (
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
            pytest.raises(_Status),
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb._half_open_probe_active is False
        admission = cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is True

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_translation_abort_in_half_open_frees_probe_slot(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status(429))
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())
        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )
        anthropic_cb = solwyn._get_circuit_breaker("anthropic")
        anthropic_cb.record_failure()
        assert anthropic_cb.state == CircuitState.OPEN

        request = {**_PLAIN_REQUEST, "n": 2}
        with (
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
            pytest.raises(UntranslatableRequestError),
        ):
            await solwyn.chat.completions.create(**request)

        anthropic.messages.create.assert_not_called()
        assert anthropic_cb.state == CircuitState.HALF_OPEN
        assert anthropic_cb._half_open_probe_active is False
        admission = anthropic_cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is True

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()


# ── circuit-breaker lazy creation under sync concurrency ─────────────────


@pytest.mark.unit
class TestCircuitBreakerLazyCreation:
    def test_get_circuit_breaker_lazy_create_is_atomic(self) -> None:
        solwyn = _make_solwyn(_openai_client(), model="gpt-5.5")
        solwyn._circuit_breakers.clear()
        created: list[object] = []
        original_new_breaker = solwyn._new_circuit_breaker

        def slow_new_breaker() -> object:
            time.sleep(0.02)
            breaker = original_new_breaker()
            created.append(breaker)
            return breaker

        with (
            patch.object(solwyn, "_new_circuit_breaker", side_effect=slow_new_breaker),
            ThreadPoolExecutor(max_workers=12) as pool,
        ):
            breakers = list(pool.map(lambda _i: solwyn._get_circuit_breaker("google"), range(12)))

        assert len({id(breaker) for breaker in breakers}) == 1
        assert len(created) == 1

        _close(solwyn)
