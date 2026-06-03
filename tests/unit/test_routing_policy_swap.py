"""P5 DoD: injectable SelectionPolicy + latency/cost routing signals.

The headline P5 invariant is the POLICY-SWAP DROP-IN: swapping the injected
``selection_policy`` changes routing order with ZERO changes to dispatch /
translation / budget. These tests build real ``Solwyn`` clients over the same
provider chain and assert ``_select_candidates`` reorders purely as a function
of the injected policy and the routing SIGNALS (observed p50, server price
hint), never touching the dispatch loop.

Also covers the two signal seams that feed those policies:
- latency observation: ``record_latency`` / ``observed_p50`` (min-sample
  threshold, correct median, thread-safety smoke), surfaced onto the candidate.
- price-hint plumbing: a ``BudgetCheckResponse.price_hints`` flows through
  ``BudgetCheckResult`` -> ``update_price_hints`` -> ``self._last_price_hints``
  -> ``ProviderCandidate.price_hint``, and ``CostPolicy`` orders by it. The SDK
  performs NO price arithmetic — the only price input is the server hint.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._routing import (
    CostPolicy,
    HealthBasedPolicy,
    LatencyPolicy,
    ProviderCandidate,
    RoutingRequest,
)
from solwyn._types import BudgetCheckResponse, BudgetMode, CircuitState, ProviderName
from solwyn.budget import _BudgetEnforcerBase
from solwyn.client import Solwyn

# ── client fakes (adapter detection keys off __class__.__module__) ───────────


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


def _make_solwyn(client: object, **overrides: object) -> Solwyn:
    """Build a real Solwyn with the reporter thread stopped (no I/O)."""
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


def _req() -> RoutingRequest:
    return RoutingRequest(requested_provider=ProviderName.OPENAI)


def _ordered_providers(solwyn: Solwyn) -> list[str]:
    """Attempt-order provider names from _select_candidates (the routing seam)."""
    return [rt.adapter.name for rt in solwyn._select_candidates(_req())]


# ── 1+2: POLICY-SWAP DROP-IN (the headline DoD) ──────────────────────────────


@pytest.mark.unit
def test_default_policy_is_health_based_when_none_injected() -> None:
    # Arrange — no selection_policy injected
    solwyn = _make_solwyn(
        _openai_client(),
        model="gpt-4o",
        fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
    )
    try:
        # Assert — health policy is the default; configured order preserved
        assert isinstance(solwyn._policy, HealthBasedPolicy)
        assert _ordered_providers(solwyn) == ["openai", "anthropic"]
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_latency_policy_swap_changes_order_zero_dispatch_change() -> None:
    # Arrange — SAME provider chain for both clients; the ONLY difference is the
    # injected policy. Seed differing observed p50 so anthropic is the faster.
    chain = dict(model="gpt-4o", fallback=[(_anthropic_client(), "claude-3-5-sonnet")])

    health = _make_solwyn(_openai_client(), **chain)
    latency = _make_solwyn(_openai_client(), selection_policy=LatencyPolicy(), **chain)
    try:
        # Seed identical latency samples into BOTH so the only behavioural
        # difference is the policy, not the signal store: openai slow, anthropic fast.
        for s in (health, latency):
            for _ in range(5):
                s.record_latency("openai", 400.0)
                s.record_latency("anthropic", 20.0)

        health_order = _ordered_providers(health)
        latency_order = _ordered_providers(latency)

        # Assert — default keeps configured order; LatencyPolicy reorders by p50.
        assert health_order == ["openai", "anthropic"]
        assert latency_order == ["anthropic", "openai"]
        assert health_order != latency_order

        # Drop-in proof: only the policy object differs between the two clients.
        assert isinstance(health._policy, HealthBasedPolicy)
        assert isinstance(latency._policy, LatencyPolicy)
    finally:
        _close(health)
        _close(latency)


@pytest.mark.unit
def test_cost_policy_swap_changes_order_via_price_hints() -> None:
    # Arrange — same chain; CostPolicy injected on one. Anthropic is cheaper.
    chain = dict(model="gpt-4o", fallback=[(_anthropic_client(), "claude-3-5-sonnet")])

    health = _make_solwyn(_openai_client(), **chain)
    cost = _make_solwyn(_openai_client(), selection_policy=CostPolicy(), **chain)
    try:
        # Feed the SAME server price hints to both (only the policy consumes them).
        for s in (health, cost):
            s.update_price_hints({"openai": 10.0, "anthropic": 2.0})

        health_order = _ordered_providers(health)
        cost_order = _ordered_providers(cost)

        # Assert — health ignores hints (configured order); cost reorders by hint.
        assert health_order == ["openai", "anthropic"]
        assert cost_order == ["anthropic", "openai"]
        assert health_order != cost_order
    finally:
        _close(health)
        _close(cost)


# ── 3: LATENCY OBSERVATION (record_latency / observed_p50) ───────────────────


@pytest.mark.unit
def test_observed_p50_none_below_min_samples() -> None:
    solwyn = _make_solwyn(_openai_client(), model="gpt-4o")
    try:
        # Arrange — fewer than _LATENCY_MIN_SAMPLES (3) samples
        solwyn.record_latency("openai", 100.0)
        solwyn.record_latency("openai", 200.0)

        # Assert — under-sampled -> None (never jumps the LatencyPolicy queue)
        assert solwyn.observed_p50("openai") is None
        # Unknown provider with no samples is also None
        assert solwyn.observed_p50("anthropic") is None
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_observed_p50_returns_median_above_threshold() -> None:
    solwyn = _make_solwyn(_openai_client(), model="gpt-4o")
    try:
        # Arrange — odd count -> exact median is the middle value
        for ms in (10.0, 50.0, 30.0):
            solwyn.record_latency("openai", ms)

        # Assert
        assert solwyn.observed_p50("openai") == 30.0

        # Even count -> statistics.median averages the two middle values
        solwyn.record_latency("openai", 70.0)  # samples: 10,50,30,70 -> median 40
        assert solwyn.observed_p50("openai") == 40.0
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_record_latency_thread_safety_smoke() -> None:
    solwyn = _make_solwyn(_openai_client(), model="gpt-4o")
    try:
        # Arrange — many threads hammering record_latency concurrently
        n_threads = 8
        per_thread = 200
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(per_thread):
                solwyn.record_latency("openai", 100.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert — no corruption: window is capped at _LATENCY_WINDOW and the
        # median of an all-100.0 window is exactly 100.0 (no torn writes).
        from solwyn._base import _LATENCY_WINDOW

        with solwyn._latency_lock:
            window_len = len(solwyn._latency_windows["openai"])
        assert window_len == _LATENCY_WINDOW
        assert solwyn.observed_p50("openai") == 100.0
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_select_candidates_surfaces_observed_p50_onto_candidate() -> None:
    # Arrange — capture the candidates the policy receives to inspect latency_p50.
    captured: list[object] = []

    class _CapturePolicy:
        def order(self, candidates, req):  # type: ignore[no-untyped-def]
            captured.extend(candidates)
            return list(candidates)

    solwyn = _make_solwyn(
        _openai_client(),
        model="gpt-4o",
        fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        selection_policy=_CapturePolicy(),
    )
    try:
        for _ in range(3):
            solwyn.record_latency("openai", 123.0)

        solwyn._select_candidates(_req())

        # Assert — observed p50 is surfaced onto the openai candidate; anthropic
        # (no samples) carries None.
        by_provider = {c.runtime.adapter.name: c for c in captured}  # type: ignore[attr-defined]
        assert by_provider["openai"].latency_p50 == 123.0  # type: ignore[attr-defined]
        assert by_provider["anthropic"].latency_p50 is None  # type: ignore[attr-defined]
    finally:
        _close(solwyn)


# ── 4: PRICE-HINT PLUMBING (server hint -> result -> candidate -> CostPolicy) ─


@pytest.mark.unit
def test_price_hints_flow_from_response_to_result() -> None:
    # Arrange — a real BudgetCheckResponse carrying server price hints.
    response = BudgetCheckResponse(
        allowed=True,
        remaining_budget=80.0,
        reservation_id="res_1",
        mode=BudgetMode.ALERT_ONLY,
        budget_limit=100.0,
        current_usage=20.0,
        denied_by_period=None,
        project_id="proj_x",
        price_hints={ProviderName.OPENAI: 10.0, ProviderName.ANTHROPIC: 2.0},
    )
    enforcer = _BudgetEnforcerBase(api_url="http://x", api_key="k")

    # Act — the sans-I/O result builder forwards the server signal verbatim,
    # re-keyed by provider VALUE for the routing layer.
    result = enforcer._build_result_from_response(response)

    # Assert
    assert result.price_hints == {"openai": 10.0, "anthropic": 2.0}


@pytest.mark.unit
def test_price_hints_plumb_into_last_price_hints_and_candidate() -> None:
    solwyn = _make_solwyn(
        _openai_client(),
        model="gpt-4o",
        fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        selection_policy=CostPolicy(),
    )
    try:
        # Act — the client calls update_price_hints after each budget check; here
        # we drive that seam directly with a server-shaped hint dict.
        solwyn.update_price_hints({"openai": 10.0, "anthropic": 2.0})

        # Assert — stored on the instance...
        assert solwyn._last_price_hints == {"openai": 10.0, "anthropic": 2.0}

        # ...surfaced onto each candidate and consumed by CostPolicy to reorder.
        ordered = _ordered_providers(solwyn)
        assert ordered == ["anthropic", "openai"]
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_cost_routing_does_no_price_arithmetic_only_server_hint() -> None:
    # Arrange — the ONLY price input is the server hint. We prove ordering is a
    # pure relative comparison: swapping which provider has the lower hint flips
    # the order, and no token count or rate is ever multiplied in.
    solwyn = _make_solwyn(
        _openai_client(),
        model="gpt-4o",
        fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        selection_policy=CostPolicy(),
    )
    try:
        # openai cheaper -> openai first
        solwyn.update_price_hints({"openai": 1.0, "anthropic": 9.0})
        assert _ordered_providers(solwyn) == ["openai", "anthropic"]

        # flip the hints -> order flips, with NO other input changed (same tokens)
        solwyn.update_price_hints({"openai": 9.0, "anthropic": 1.0})
        assert _ordered_providers(solwyn) == ["anthropic", "openai"]

        # estimated_input_tokens on the request is NOT a price input: changing it
        # must not change cost order (proves no per-token price math).
        ordered_a = [
            rt.adapter.name
            for rt in solwyn._select_candidates(
                RoutingRequest(requested_provider=ProviderName.OPENAI, estimated_input_tokens=1)
            )
        ]
        ordered_b = [
            rt.adapter.name
            for rt in solwyn._select_candidates(
                RoutingRequest(
                    requested_provider=ProviderName.OPENAI, estimated_input_tokens=1_000_000
                )
            )
        ]
        assert ordered_a == ordered_b == ["anthropic", "openai"]
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_no_price_hints_leaves_cost_policy_on_health_order() -> None:
    # Arrange — CostPolicy injected but the server has provided NO hints yet.
    solwyn = _make_solwyn(
        _openai_client(),
        model="gpt-4o",
        fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        selection_policy=CostPolicy(),
    )
    try:
        # Assert — falls back to health/config order (identical to today).
        assert solwyn._last_price_hints == {}
        assert _ordered_providers(solwyn) == ["openai", "anthropic"]
    finally:
        _close(solwyn)


# ── 5: defensive validation of a misbehaving policy's output (fix F) ─────────


@pytest.mark.unit
def test_select_candidates_drops_foreign_runtime_from_misbehaving_policy() -> None:
    # A custom SelectionPolicy must not be able to inject a runtime that was never
    # in the configured chain into the dispatch walk. _select_candidates keeps
    # only candidates whose runtime is one of self._runtimes, preserving the
    # policy's order for the valid subset.
    class _ForeignAppendingPolicy:
        """Returns the real candidates THEN appends a fabricated foreign one."""

        def order(self, candidates, req):  # type: ignore[no-untyped-def]
            foreign_runtime = MagicMock()
            foreign_runtime.adapter.name = "evil-provider"
            foreign = ProviderCandidate(
                runtime=foreign_runtime,
                breaker_state=CircuitState.CLOSED,
                recovery_eligible=True,
                translatable=True,
            )
            # Foreign candidate placed FIRST to prove ordering of the valid subset
            # is preserved after the foreign one is dropped (not just appended-and-cut).
            return [foreign, *candidates]

    solwyn = _make_solwyn(
        _openai_client(),
        model="gpt-4o",
        fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        selection_policy=_ForeignAppendingPolicy(),
    )
    try:
        ordered = solwyn._select_candidates(_req())
        names = [rt.adapter.name for rt in ordered]

        # Assert — the fabricated foreign runtime is dropped; only chain runtimes
        # survive, and every survivor is an identity from self._runtimes.
        assert "evil-provider" not in names
        assert names == ["openai", "anthropic"]
        assert all(rt in solwyn._runtimes for rt in ordered)
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_select_candidates_preserves_policy_order_for_valid_subset() -> None:
    # When a policy reorders the REAL candidates (and also slips in a foreign one),
    # the valid subset keeps the policy's chosen order, not the configured order.
    class _ReverseWithForeignPolicy:
        def order(self, candidates, req):  # type: ignore[no-untyped-def]
            foreign_runtime = MagicMock()
            foreign_runtime.adapter.name = "ghost"
            foreign = ProviderCandidate(
                runtime=foreign_runtime,
                breaker_state=CircuitState.CLOSED,
                recovery_eligible=True,
                translatable=True,
            )
            # Reverse the real candidates AND inject a foreign one in the middle.
            reversed_real = list(reversed(candidates))
            return [reversed_real[0], foreign, *reversed_real[1:]]

    solwyn = _make_solwyn(
        _openai_client(),
        model="gpt-4o",
        fallback=[(_anthropic_client(), "claude-3-5-sonnet")],
        selection_policy=_ReverseWithForeignPolicy(),
    )
    try:
        names = [rt.adapter.name for rt in solwyn._select_candidates(_req())]

        # Assert — foreign dropped; the policy's REVERSED order of the valid
        # subset is preserved (anthropic before openai).
        assert "ghost" not in names
        assert names == ["anthropic", "openai"]
    finally:
        _close(solwyn)
