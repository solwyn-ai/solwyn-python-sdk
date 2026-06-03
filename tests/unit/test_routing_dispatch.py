"""P5 routing signals proven END-TO-END through the real dispatch loop.

Where ``test_routing_policy_swap.py`` drives the ``_select_candidates`` seam
directly, these tests run real intercepted calls (mocked provider clients) so
the wiring between dispatch and the routing signals is exercised for real:

- price-hint LIFETIME (fix A): hints persist across a budget cache hit; a
  hint-less (None) budget response must NOT wipe the last-known hints.
- POLICY-SWAP DROP-IN through dispatch (fix B): the SAME provider chain, the
  SAME dispatch/translation/budget code, served provider changes purely from
  the injected ``selection_policy``.
- latency observation wiring (fix C): successful non-streaming AND streamed
  hops feed ``record_latency`` so ``observed_p50`` transitions None -> value.
- price-hint plumbing INTO ``update_price_hints`` (fix D): a budget check that
  returns non-None ``price_hints`` populates ``_last_price_hints`` and a
  subsequent ``CostPolicy`` ordering reflects it.

No real provider SDKs are importable; clients are duck-typed MagicMocks whose
``__class__.__module__`` triggers adapter detection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._routing import CostPolicy, LatencyPolicy, RoutingRequest
from solwyn._types import ProviderName
from solwyn.budget import BudgetCheckResult
from solwyn.client import AsyncSolwyn, Solwyn

# ── client + response fakes ──────────────────────────────────────────────


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
    message = SimpleNamespace(role="assistant", content="ok from gpt", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_response() -> SimpleNamespace:
    block = SimpleNamespace(type="text", text="ok from claude")
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        model="claude-3-5-sonnet",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _openai_text_chunk(text: str | None, finish: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(role=None, content=text, tool_calls=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], model="gpt-4o", usage=None)


def _openai_usage_chunk() -> SimpleNamespace:
    # Terminal usage-only chunk so the accumulator settles and on_complete fires.
    return SimpleNamespace(
        choices=[],
        model="gpt-4o",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


def _budget_result(
    *, reservation_id: str | None = None, price_hints: dict[str, float] | None = None
) -> BudgetCheckResult:
    """A real allow-result; ``price_hints=None`` models the cache-hit path."""
    return BudgetCheckResult(
        allowed=True,
        remaining_budget=80.0,
        reservation_id=reservation_id,
        price_hints=price_hints,
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
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
    # max_tokens so a cross-provider hop translates cleanly (Anthropic requires it).
    "max_tokens": 256,
}


def _ordered_providers(solwyn: Solwyn) -> list[str]:
    req = RoutingRequest(requested_provider=ProviderName.OPENAI)
    return [rt.adapter.name for rt in solwyn._select_candidates(req)]


# ── A: price-hint LIFETIME across a budget cache hit ─────────────────────


@pytest.mark.unit
class TestPriceHintLifetimeAcrossCacheHit:
    def test_cache_hit_with_none_hints_preserves_last_known_hints(self) -> None:
        # Arrange — two successive intercepted calls. The FIRST budget response
        # carries server price hints; the SECOND is a cache hit (price_hints
        # None). The advisory hints must SURVIVE the hint-less second check.
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        )
        try:
            results = [
                _budget_result(price_hints={"openai": 10.0, "anthropic": 2.0}),
                _budget_result(price_hints=None),  # cache hit
            ]
            with patch.object(solwyn._budget, "check_budget", side_effect=results):
                solwyn.chat.completions.create(**_PLAIN_REQUEST)
                # After the first (hint-bearing) call the signal is populated.
                assert solwyn._last_price_hints == {"openai": 10.0, "anthropic": 2.0}

                solwyn.chat.completions.create(**_PLAIN_REQUEST)

            # Assert — the cache-hit (None) response did NOT wipe the hints; the
            # CostPolicy signal is intact for the whole cache window.
            assert solwyn._last_price_hints == {"openai": 10.0, "anthropic": 2.0}
        finally:
            _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_cache_hit_with_none_hints_preserves_last_known_hints(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(return_value=_openai_response())
        solwyn = AsyncSolwyn(  # type: ignore[arg-type]
            openai,
            api_key=VALID_API_KEY,
            model="gpt-4o",
            fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        )
        solwyn._reporter.report = MagicMock()
        try:
            results = [
                _budget_result(price_hints={"openai": 10.0, "anthropic": 2.0}),
                _budget_result(price_hints=None),
            ]
            with patch.object(solwyn._budget, "check_budget", AsyncMock(side_effect=results)):
                await solwyn.chat.completions.create(**_PLAIN_REQUEST)
                assert solwyn._last_price_hints == {"openai": 10.0, "anthropic": 2.0}
                await solwyn.chat.completions.create(**_PLAIN_REQUEST)

            assert solwyn._last_price_hints == {"openai": 10.0, "anthropic": 2.0}
        finally:
            await solwyn._reporter._http.aclose()
            await solwyn._budget._http.aclose()


# ── B: POLICY-SWAP DROP-IN through the real dispatch loop ────────────────


@pytest.mark.unit
class TestPolicySwapThroughDispatch:
    def _chain(self) -> dict[str, Any]:
        # SAME chain for every variant — only selection_policy will differ.
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        return {
            "primary": openai,
            "fallback_client": anthropic,
            "chain": dict(model="gpt-4o", fallback=[(anthropic, "claude-3-5-sonnet")]),
        }

    def _served_provider(self, c: dict[str, Any], **policy_kwargs: object) -> str:
        solwyn = _make_solwyn(c["primary"], **{**c["chain"], **policy_kwargs})
        try:
            # Seed latency so anthropic is the FASTER provider (LatencyPolicy
            # would route to it); openai stays the configured primary.
            for _ in range(5):
                solwyn.record_latency("openai", 400.0)
                solwyn.record_latency("anthropic", 20.0)
            with patch.object(solwyn._budget, "check_budget", return_value=_budget_result()):
                result = solwyn.chat.completions.create(**_PLAIN_REQUEST)
            # OpenAI primary returns native; an Anthropic served hop is normalized
            # back to OpenAI dialect but its text is "ok from claude".
            served_text = result.choices[0].message.content
            return "anthropic" if served_text == "ok from claude" else "openai"
        finally:
            _close(solwyn)

    def test_served_provider_changes_purely_from_injected_policy(self) -> None:
        # Health default keeps configured order -> OPENAI serves.
        health_served = self._served_provider(self._chain())
        # LatencyPolicy (anthropic faster) reorders -> ANTHROPIC serves, with
        # identical dispatch/translation/budget; only the policy arg differs.
        latency_served = self._served_provider(self._chain(), selection_policy=LatencyPolicy())

        assert health_served == "openai"
        assert latency_served == "anthropic"
        assert health_served != latency_served

    def test_cost_policy_variant_routes_by_price_hint_through_dispatch(self) -> None:
        # CostPolicy with anthropic cheaper -> ANTHROPIC serves, same dispatch.
        c = self._chain()
        solwyn = _make_solwyn(c["primary"], selection_policy=CostPolicy(), **c["chain"])
        try:
            with patch.object(
                solwyn._budget,
                "check_budget",
                return_value=_budget_result(price_hints={"openai": 9.0, "anthropic": 1.0}),
            ):
                result = solwyn.chat.completions.create(**_PLAIN_REQUEST)
            assert result.choices[0].message.content == "ok from claude"
        finally:
            _close(solwyn)


# ── C: latency observation wired into the dispatch success paths ─────────


@pytest.mark.unit
class TestLatencyObservedThroughDispatch:
    def test_non_streaming_successes_transition_observed_p50(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-4o")
        try:
            # Below the min-sample threshold -> None.
            assert solwyn.observed_p50("openai") is None

            with patch.object(solwyn._budget, "check_budget", return_value=_budget_result()):
                for _ in range(4):
                    solwyn.chat.completions.create(**_PLAIN_REQUEST)

            # After enough successful non-streaming hops, record_latency has been
            # driven from the dispatch success path -> a real p50 appears.
            p50 = solwyn.observed_p50("openai")
            assert p50 is not None
            assert p50 >= 0.0
        finally:
            _close(solwyn)

    def test_streamed_call_drained_to_on_complete_records_latency(self) -> None:
        openai = _openai_client()
        # Each streamed call returns a fresh iterator settling on a usage chunk.
        openai.chat.completions.create.side_effect = lambda **_kw: iter(
            [_openai_text_chunk("hi", finish="stop"), _openai_usage_chunk()]
        )
        solwyn = _make_solwyn(openai, model="gpt-4o")
        try:
            with patch.object(solwyn._budget, "check_budget", return_value=_budget_result()):
                for _ in range(3):
                    stream = solwyn.chat.completions.create(stream=True, **_PLAIN_REQUEST)
                    # Drain fully so the wrapper fires on_complete (latency record).
                    list(stream)

            # on_complete recorded the served-provider latency on each settle.
            p50 = solwyn.observed_p50("openai")
            assert p50 is not None
        finally:
            _close(solwyn)


# ── D: price-hint plumbing FROM the budget check INTO update_price_hints ──


@pytest.mark.unit
class TestPriceHintPlumbingFromBudgetCheck:
    def test_budget_hints_populate_last_price_hints_and_drive_cost_order(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet")],
            selection_policy=CostPolicy(),
        )
        try:
            # Before any check, no hints -> CostPolicy is on health/config order.
            assert solwyn._last_price_hints == {}
            assert _ordered_providers(solwyn) == ["openai", "anthropic"]

            with patch.object(
                solwyn._budget,
                "check_budget",
                return_value=_budget_result(price_hints={"openai": 10.0, "anthropic": 2.0}),
            ):
                solwyn.chat.completions.create(**_PLAIN_REQUEST)

            # The budget check's hints flowed into _last_price_hints...
            assert solwyn._last_price_hints == {"openai": 10.0, "anthropic": 2.0}
            # ...and a subsequent CostPolicy ordering reflects them (anthropic
            # cheaper -> first).
            assert _ordered_providers(solwyn) == ["anthropic", "openai"]
        finally:
            _close(solwyn)
