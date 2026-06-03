"""Cross-provider failover routing — the P1 DoD (spec §10 P1, §4.6).

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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._types import CircuitState
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ProviderUnavailableError


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


def _openai_response() -> SimpleNamespace:
    # Native OpenAI Chat Completions shape (duck-typed): the adapter reads
    # ``usage`` and the normalizer reads ``choices[0].message`` + ``model``.
    message = SimpleNamespace(role="assistant", content="ok from gpt", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-4o",
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
        model="claude-3-5-sonnet",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _allow_budget() -> SimpleNamespace:
    """Allow result with no reservation (skips confirm_cost)."""
    return SimpleNamespace(allowed=True, reservation_id=None)


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
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
}


# ── cross-provider failover (the core P1 DoD case) ───────────────────────


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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Anthropic served the request. The plain-text request crossed via real
        # §5 translation, and the served response was normalized back into the
        # OpenAI dialect the caller wrote — so the native access path resolves
        # to the Anthropic text (NOT identity with the raw Anthropic object).
        assert result is not anthropic_resp
        assert result.choices[0].message.content == "ok from claude"
        anthropic.messages.create.assert_called_once()
        openai.chat.completions.create.assert_called_once()
        # The Anthropic call received TRANSLATED, Anthropic-native kwargs.
        anthropic_kwargs = anthropic.messages.create.call_args.kwargs
        assert anthropic_kwargs["max_tokens"] == 256  # from entry default_params
        assert anthropic_kwargs["model"] == "claude-3-5-sonnet"  # fallback entry model
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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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
        # failover_reason for the dashboard (§4.6, §8.2/§8.5). Here the primary
        # is CLOSED, ATTEMPTED, and 429s — a REACTIVE failover -> PRIMARY_ERROR.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        ev = success[0]
        assert ev.provider.value == "anthropic"  # served, not requested
        assert ev.model == "claude-3-5-sonnet"
        assert ev.is_provider_fallback is True
        assert ev.is_model_fallback is False
        assert ev.requested_provider.value == "openai"
        assert ev.requested_model == "gpt-4o"
        # Primary was attempted-and-errored -> reactive failover -> PRIMARY_ERROR.
        assert ev.failover_reason is not None and ev.failover_reason.value == "primary_error"
        assert ev.attempt_index == 1

        _close(solwyn)

    def test_primary_errored_success_reason_is_primary_error(self) -> None:
        # §8.2/§8.5 [A]: PRIMARY CLOSED, ATTEMPTED in this walk, raises 429 ->
        # the cross-provider success event's failover_reason is PRIMARY_ERROR
        # (reactive failover), NOT CIRCUIT_OPEN.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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

        # §8.2/§8.5 [A]: the primary was SKIPPED (breaker pre-OPEN), never
        # attempted in this walk -> proactive reroute -> CIRCUIT_OPEN.
        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        assert success[0].is_provider_fallback is True
        assert success[0].failover_reason is not None
        assert success[0].failover_reason.value == "circuit_open"

        _close(solwyn)

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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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
        assert anthropic_kwargs["model"] == "claude-3-5-sonnet"
        # Breaker accounting mirrors the sync case.
        assert openai_cb.failure_count == 1
        assert anthropic_cb.failure_count == 0

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ── same-provider model swap ─────────────────────────────────────────────


@pytest.mark.unit
class TestSameProviderModelSwap:
    def test_model_swap_on_same_client(self) -> None:
        # Primary model fails (429); the SAME client serves the gpt-4o-mini swap.
        client = _openai_client()
        success = _openai_response()
        client.chat.completions.create.side_effect = [_Status(429), success]

        solwyn = _make_solwyn(client, model="gpt-4o", fallback=[(client, "gpt-4o-mini")])
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"

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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # A 400 classifies FAIL_FAST: the chain STOPS, the fallback is NOT
        # attempted, and the ORIGINAL exception propagates unchanged (§6.1).
        assert exc_info.value is original
        anthropic.messages.create.assert_not_called()
        openai.chat.completions.create.assert_called_once()

        # §6.1 disposition table: FAIL_FAST is request-shaped, NOT a provider-
        # health signal, so it must NOT count the breaker.
        assert openai_cb.failure_count == 0
        assert openai_cb.state == CircuitState.CLOSED

        _close(solwyn)

    def test_post_send_ambiguous_counts_breaker_but_does_not_failover(self) -> None:
        # A 500 classifies POST_SEND_AMBIGUOUS: it IS a health signal (counts the
        # breaker) but the request may have run, so the chain must NOT fail over
        # under the default "safe" idempotency — it re-raises the original (§6.1).
        openai = _openai_client()
        original = _Status(500, "server error")
        openai.chat.completions.create.side_effect = original
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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


# ── per-hop deadline bound ───────────────────────────────────────────────


@pytest.mark.unit
class TestPerHopDeadline:
    def test_with_options_called_with_max_retries_zero_and_finite_timeout(self) -> None:
        client = _openai_client()
        client.chat.completions.create.return_value = _openai_response()

        solwyn = _make_solwyn(client, model="gpt-4o")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        client.with_options.assert_called_once()
        call = client.with_options.call_args
        assert call.kwargs["max_retries"] == 0
        timeout = call.kwargs["timeout"]
        assert isinstance(timeout, int | float)
        assert 0.0 < timeout < float("inf")

        _close(solwyn)

    def test_per_hop_timeout_shrinks_across_candidates(self) -> None:
        # §6.3: per-hop timeout = remaining / (candidates not yet attempted).
        # With a 30s chain budget and 2 candidates, hop0 gets ~total/2 (NOT the
        # full remaining); the final hop gets the whole remaining slice.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            failover_total_timeout=30.0,
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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
        # carries the full attempted chain (§6.3).
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            failover_total_timeout=0.0,
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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
