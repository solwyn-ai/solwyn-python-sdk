"""Routing signals proven END-TO-END through the real dispatch loop.

Where ``test_routing_policy_swap.py`` drives the ``_select_candidates`` seam
directly, these tests run real intercepted calls (mocked provider clients) so
the wiring between dispatch and the routing signals is exercised for real:

- POLICY-SWAP DROP-IN through dispatch (fix B): the SAME provider chain, the
  SAME dispatch/translation/budget code, served provider changes purely from
  the injected ``selection_policy``.
- latency observation wiring (fix C): successful non-streaming AND streamed
  hops feed ``record_latency`` so ``observed_p50`` transitions None -> value.
- request-scoped price-hint plumbing (fix D): each budget check's
  ``price_hints`` orders only that dispatch and never affects a following call.

No real provider SDKs are importable; clients are duck-typed MagicMocks whose
``__class__.__module__`` triggers adapter detection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._routing import CostPolicy, LatencyPolicy
from solwyn.budget import BudgetCheckResult
from solwyn.client import Solwyn

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
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_response() -> SimpleNamespace:
    block = SimpleNamespace(type="text", text="ok from claude")
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _openai_text_chunk(text: str | None, finish: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(role=None, content=text, tool_calls=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], model="gpt-5.5", usage=None)


def _openai_usage_chunk() -> SimpleNamespace:
    # Terminal usage-only chunk so the accumulator settles and on_complete fires.
    return SimpleNamespace(
        choices=[],
        model="gpt-5.5",
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
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    solwyn._solwyn_reporter.report = MagicMock()
    return solwyn


def _close(solwyn: Solwyn) -> None:
    solwyn._solwyn_reporter._http.close()
    solwyn._solwyn_budget._http.close()


_PLAIN_REQUEST = {
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "hi"}],
    # max_tokens so a cross-provider hop translates cleanly (Anthropic requires it).
    "max_tokens": 256,
}


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
            "chain": dict(model="gpt-5.5", fallback=[(anthropic, "claude-sonnet-5")]),
        }

    def _served_provider(self, c: dict[str, Any], **policy_kwargs: object) -> str:
        solwyn = _make_solwyn(c["primary"], **{**c["chain"], **policy_kwargs})
        try:
            # Seed latency so anthropic is the FASTER provider (LatencyPolicy
            # would route to it); openai stays the configured primary.
            for _ in range(5):
                solwyn.record_latency("openai", 400.0)
                solwyn.record_latency("anthropic", 20.0)
            with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_budget_result()):
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
                solwyn._solwyn_budget,
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
        solwyn = _make_solwyn(openai, model="gpt-5.5")
        try:
            # Below the min-sample threshold -> None.
            assert solwyn.observed_p50("openai") is None

            with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_budget_result()):
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
        solwyn = _make_solwyn(openai, model="gpt-5.5")
        try:
            with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_budget_result()):
                for _ in range(3):
                    stream = solwyn.chat.completions.create(stream=True, **_PLAIN_REQUEST)
                    # Drain fully so the wrapper fires on_complete (latency record).
                    list(stream)

            # on_complete recorded the served-provider latency on each settle.
            p50 = solwyn.observed_p50("openai")
            assert p50 is not None
        finally:
            _close(solwyn)


# ── D: request-scoped price-hint plumbing from the budget check ───────────


@pytest.mark.unit
class TestPriceHintPlumbingFromBudgetCheck:
    def test_budget_hints_order_only_the_matching_dispatch(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5")],
            selection_policy=CostPolicy(),
        )
        try:
            with patch.object(
                solwyn._solwyn_budget,
                "check_budget",
                side_effect=[
                    _budget_result(price_hints={"openai": 10.0, "anthropic": 2.0}),
                    _budget_result(price_hints=None),
                ],
            ):
                first = solwyn.chat.completions.create(**_PLAIN_REQUEST)
                second = solwyn.chat.completions.create(**_PLAIN_REQUEST)

            # First call receives prices making Anthropic cheaper; the next
            # check carries None, so its configured primary OpenAI is selected.
            assert first.choices[0].message.content == "ok from claude"
            assert second.choices[0].message.content == "ok from gpt"
            assert anthropic.messages.create.call_count == 1
            assert openai.chat.completions.create.call_count == 1
        finally:
            _close(solwyn)


@pytest.mark.unit
class TestDeadlineBoundsThroughDispatch:
    def test_budget_and_provider_hop_timeouts_do_not_exceed_chain_deadline(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            failover_total_timeout=0.25,
        )
        try:
            check_spy = MagicMock(return_value=_budget_result())
            with patch.object(solwyn._solwyn_budget, "check_budget", check_spy):
                solwyn.chat.completions.create(**_PLAIN_REQUEST)

            budget_timeout = check_spy.call_args.kwargs["timeout"]
            hop_timeout = openai.with_options.call_args.kwargs["timeout"].connect
            assert 0 < budget_timeout <= 0.25
            assert 0 < hop_timeout <= 0.25
        finally:
            _close(solwyn)
