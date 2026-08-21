"""Unit tests for solwyn._routing (pure, sans-I/O selection policy).

HealthBasedPolicy.order is a PURE ordering over non-mutating breaker-state
snapshots: it filters out OPEN-not-recovery-eligible candidates and ranks the
rest CLOSED < HALF_OPEN < recovery-eligible OPEN, sinking untranslatable
targets within a state tier. sorted() is stable, so configured (input) order is
preserved inside a priority tier. order() must NEVER call admit() or
mutate a breaker (inspection-vs-consumption rule).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY, make_mock_client

from solwyn._routing import (
    CostPolicy,
    HealthBasedPolicy,
    LatencyPolicy,
    ProviderCandidate,
    RoutingRequest,
)
from solwyn._types import CircuitState, ProviderName
from solwyn.client import Solwyn


def _candidate(
    state: CircuitState,
    *,
    recovery_eligible: bool = False,
    translatable: bool = True,
    tag: str = "",
    latency_p50: float | None = None,
    price_hint: float | None = None,
) -> ProviderCandidate:
    """Build a ProviderCandidate with a dummy (non-runtime) object in the runtime slot."""
    runtime = types.SimpleNamespace(tag=tag)
    return ProviderCandidate(
        runtime=runtime,  # type: ignore[arg-type]
        breaker_state=state,
        recovery_eligible=recovery_eligible,
        translatable=translatable,
        latency_p50=latency_p50,
        price_hint=price_hint,
    )


def _req() -> RoutingRequest:
    return RoutingRequest(requested_provider=ProviderName.OPENAI)


def _routing_client(base_url: str) -> MagicMock:
    client = make_mock_client()
    client.base_url = base_url
    return client


def _close(solwyn: Solwyn) -> None:
    solwyn._solwyn_reporter._http.close()
    solwyn._solwyn_budget._http.close()


@pytest.mark.unit
def test_closed_before_half_open_before_recovery_eligible_open() -> None:
    # Arrange — deliberately reversed input order
    open_eligible = _candidate(CircuitState.OPEN, recovery_eligible=True, tag="open")
    half = _candidate(CircuitState.HALF_OPEN, tag="half")
    closed = _candidate(CircuitState.CLOSED, tag="closed")

    # Act
    ordered = HealthBasedPolicy().order([open_eligible, half, closed], _req())

    # Assert
    assert [c.breaker_state for c in ordered] == [
        CircuitState.CLOSED,
        CircuitState.HALF_OPEN,
        CircuitState.OPEN,
    ]


@pytest.mark.unit
def test_open_not_recovery_eligible_is_filtered_out() -> None:
    # Arrange
    closed = _candidate(CircuitState.CLOSED, tag="closed")
    open_blocked = _candidate(CircuitState.OPEN, recovery_eligible=False, tag="blocked")
    open_eligible = _candidate(CircuitState.OPEN, recovery_eligible=True, tag="eligible")

    # Act
    ordered = HealthBasedPolicy().order([closed, open_blocked, open_eligible], _req())

    # Assert — blocked OPEN dropped entirely
    tags = [c.runtime.tag for c in ordered]
    assert tags == ["closed", "eligible"]
    assert "blocked" not in tags


@pytest.mark.unit
def test_stable_config_order_within_a_tier() -> None:
    # Arrange — three CLOSED candidates in a specific configured order
    a = _candidate(CircuitState.CLOSED, tag="a")
    b = _candidate(CircuitState.CLOSED, tag="b")
    c = _candidate(CircuitState.CLOSED, tag="c")

    # Act
    ordered = HealthBasedPolicy().order([a, b, c], _req())

    # Assert — input order preserved (stable sort)
    assert [cand.runtime.tag for cand in ordered] == ["a", "b", "c"]


@pytest.mark.unit
def test_untranslatable_sinks_below_translatable_within_same_state() -> None:
    # Arrange — untranslatable listed FIRST, both CLOSED
    untranslatable = _candidate(CircuitState.CLOSED, translatable=False, tag="no_xlate")
    translatable = _candidate(CircuitState.CLOSED, translatable=True, tag="xlate")

    # Act
    ordered = HealthBasedPolicy().order([untranslatable, translatable], _req())

    # Assert — translatable wins the tie within the CLOSED tier
    assert [c.runtime.tag for c in ordered] == ["xlate", "no_xlate"]


@pytest.mark.unit
def test_state_tier_dominates_translatability() -> None:
    # Arrange — a translatable HALF_OPEN must still sort below an untranslatable CLOSED
    closed_untranslatable = _candidate(CircuitState.CLOSED, translatable=False, tag="closed_no")
    half_translatable = _candidate(CircuitState.HALF_OPEN, translatable=True, tag="half_yes")

    # Act
    ordered = HealthBasedPolicy().order([half_translatable, closed_untranslatable], _req())

    # Assert — state tier dominates the secondary translatable key
    assert [c.runtime.tag for c in ordered] == ["closed_no", "half_yes"]


@pytest.mark.unit
def test_empty_input_returns_empty() -> None:
    # Act / Assert
    assert HealthBasedPolicy().order([], _req()) == []


@pytest.mark.unit
def test_all_open_blocked_returns_empty() -> None:
    # Arrange
    blocked1 = _candidate(CircuitState.OPEN, recovery_eligible=False, tag="b1")
    blocked2 = _candidate(CircuitState.OPEN, recovery_eligible=False, tag="b2")

    # Act / Assert
    assert HealthBasedPolicy().order([blocked1, blocked2], _req()) == []


@pytest.mark.unit
def test_order_does_not_touch_the_breaker() -> None:
    # Arrange — a breaker mock that would explode if any method/attr were read
    breaker = MagicMock()
    breaker.admit.side_effect = AssertionError("order() must not call admit()")

    runtime = types.SimpleNamespace(breaker=breaker)
    candidate = ProviderCandidate(
        runtime=runtime,  # type: ignore[arg-type]
        breaker_state=CircuitState.OPEN,
        recovery_eligible=True,
        translatable=True,
    )

    # Act
    HealthBasedPolicy().order([candidate], _req())

    # Assert — purity: the breaker was never inspected or mutated
    breaker.admit.assert_not_called()
    assert breaker.method_calls == []


@pytest.mark.unit
def test_candidate_is_frozen_and_defaults_price_hint_none() -> None:
    # Arrange
    candidate = _candidate(CircuitState.CLOSED)

    # Assert — frozen dataclass, price_hint defaults to None
    assert candidate.price_hint is None
    with pytest.raises(AttributeError):
        candidate.translatable = False  # type: ignore[misc]


@pytest.mark.unit
def test_routing_request_rejects_extra_fields() -> None:
    # Assert — the dataclass constructor preserves unexpected-kwarg rejection
    with pytest.raises(TypeError):
        RoutingRequest(requested_provider=ProviderName.OPENAI, bogus=1)  # type: ignore[call-arg]


@pytest.mark.unit
def test_routing_request_estimated_input_tokens_defaults_zero() -> None:
    # Assert
    assert RoutingRequest(requested_provider=ProviderName.ANTHROPIC).estimated_input_tokens == 0


# ── LatencyPolicy (pure drop-in over observed p50) ───────────────────────────


@pytest.mark.unit
def test_latency_policy_orders_usable_by_ascending_p50() -> None:
    # Arrange — three CLOSED candidates, deliberately not in p50 order
    slow = _candidate(CircuitState.CLOSED, tag="slow", latency_p50=300.0)
    fast = _candidate(CircuitState.CLOSED, tag="fast", latency_p50=50.0)
    mid = _candidate(CircuitState.CLOSED, tag="mid", latency_p50=120.0)

    # Act
    ordered = LatencyPolicy().order([slow, fast, mid], _req())

    # Assert — lower p50 first
    assert [c.runtime.tag for c in ordered] == ["fast", "mid", "slow"]


@pytest.mark.unit
def test_latency_policy_none_p50_sorts_last() -> None:
    # Arrange — one candidate has no observed p50 yet (under-sampled)
    unknown = _candidate(CircuitState.CLOSED, tag="unknown", latency_p50=None)
    known = _candidate(CircuitState.CLOSED, tag="known", latency_p50=200.0)

    # Act — unknown listed FIRST in config order
    ordered = LatencyPolicy().order([unknown, known], _req())

    # Assert — a known p50 (even a high one) outranks an unknown p50 in its tier
    assert [c.runtime.tag for c in ordered] == ["known", "unknown"]


@pytest.mark.unit
def test_latency_policy_preserves_health_filtering() -> None:
    # Arrange — an OPEN-not-eligible candidate with a tiny p50 must STILL be
    # dropped; latency never overrides health. CLOSED ranks before HALF_OPEN.
    open_blocked = _candidate(
        CircuitState.OPEN, recovery_eligible=False, tag="blocked", latency_p50=1.0
    )
    half_fast = _candidate(CircuitState.HALF_OPEN, tag="half", latency_p50=10.0)
    closed_slow = _candidate(CircuitState.CLOSED, tag="closed", latency_p50=999.0)

    # Act
    ordered = LatencyPolicy().order([open_blocked, half_fast, closed_slow], _req())

    # Assert — blocked dropped; health tier dominates p50 (CLOSED before HALF_OPEN)
    tags = [c.runtime.tag for c in ordered]
    assert tags == ["closed", "half"]
    assert "blocked" not in tags


@pytest.mark.unit
def test_latency_policy_health_tier_dominates_p50() -> None:
    # Arrange — a fast HALF_OPEN must still sort below a slow CLOSED: the health
    # tier is the dominant key, p50 only breaks ties WITHIN a tier.
    closed_slow = _candidate(CircuitState.CLOSED, tag="closed_slow", latency_p50=500.0)
    half_fast = _candidate(CircuitState.HALF_OPEN, tag="half_fast", latency_p50=5.0)

    # Act
    ordered = LatencyPolicy().order([half_fast, closed_slow], _req())

    # Assert
    assert [c.runtime.tag for c in ordered] == ["closed_slow", "half_fast"]


@pytest.mark.unit
def test_latency_policy_is_pure_does_not_touch_breaker() -> None:
    # Arrange — a breaker mock that explodes on any access
    breaker = MagicMock()
    breaker.admit.side_effect = AssertionError("order() must not call admit()")
    runtime = types.SimpleNamespace(breaker=breaker)
    candidate = ProviderCandidate(
        runtime=runtime,  # type: ignore[arg-type]
        breaker_state=CircuitState.CLOSED,
        recovery_eligible=True,
        translatable=True,
        latency_p50=42.0,
    )

    # Act
    LatencyPolicy().order([candidate], _req())

    # Assert — purity
    breaker.admit.assert_not_called()
    assert breaker.method_calls == []


# ── CostPolicy (pure drop-in over SERVER price hint; NO price math) ───────────


@pytest.mark.unit
def test_cost_policy_orders_usable_by_ascending_price_hint() -> None:
    # Arrange — three CLOSED candidates, not in price order
    pricey = _candidate(CircuitState.CLOSED, tag="pricey", price_hint=9.0)
    cheap = _candidate(CircuitState.CLOSED, tag="cheap", price_hint=1.0)
    mid = _candidate(CircuitState.CLOSED, tag="mid", price_hint=4.0)

    # Act
    ordered = CostPolicy().order([pricey, cheap, mid], _req())

    # Assert — lower price hint first
    assert [c.runtime.tag for c in ordered] == ["cheap", "mid", "pricey"]


@pytest.mark.unit
def test_cost_policy_none_price_sorts_last() -> None:
    # Arrange — one candidate carries no server hint
    unknown = _candidate(CircuitState.CLOSED, tag="unknown", price_hint=None)
    known = _candidate(CircuitState.CLOSED, tag="known", price_hint=7.0)

    # Act — unknown listed FIRST
    ordered = CostPolicy().order([unknown, known], _req())

    # Assert — a known hint outranks a missing hint in the same tier
    assert [c.runtime.tag for c in ordered] == ["known", "unknown"]


@pytest.mark.unit
def test_select_candidates_orders_api_ratios_and_sinks_an_unhinted_provider() -> None:
    """Catches routing that ignores a zero ratio or lets an unhinted runtime outrank hints."""
    openai = _routing_client("https://api.openai.com/v1")
    together = _routing_client("https://api.together.xyz/v1")
    anthropic = make_mock_client(module="anthropic._client", name="Anthropic")
    ollama = _routing_client("http://localhost:11434/v1")
    unhinted = _routing_client("https://api.some-new-vendor.example/v1")
    wire_price_hints = {
        "together": 1.0,
        "openai": 4.7,
        "anthropic": 9.3,
        "ollama": 0.0,
    }
    expected_hinted = [
        provider for provider, _ in sorted(wire_price_hints.items(), key=lambda item: item[1])
    ]

    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.4-nano",
            fallback=[
                (anthropic, "claude-haiku-4-5"),
                (unhinted, "vendor/model"),
                (together, "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                (ollama, "llama3.2"),
            ],
            selection_policy=CostPolicy(),
        )
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    try:
        ordered = solwyn._select_candidates(_req(), price_hints=wire_price_hints)

        assert [runtime.adapter.name for runtime in ordered] == [
            *expected_hinted,
            "openai_compatible",
        ]
    finally:
        _close(solwyn)


@pytest.mark.unit
def test_cost_policy_falls_back_to_health_order_when_no_hints() -> None:
    # Arrange — NO candidate carries a price hint -> identical to HealthBasedPolicy.
    # Deliberately reversed health order to prove the fallback really sorts.
    open_eligible = _candidate(CircuitState.OPEN, recovery_eligible=True, tag="open")
    half = _candidate(CircuitState.HALF_OPEN, tag="half")
    closed = _candidate(CircuitState.CLOSED, tag="closed")
    candidates = [open_eligible, half, closed]

    # Act
    cost_ordered = CostPolicy().order(candidates, _req())
    health_ordered = HealthBasedPolicy().order(candidates, _req())

    # Assert — byte-identical to the health-only order
    assert [c.runtime.tag for c in cost_ordered] == [c.runtime.tag for c in health_ordered]
    assert [c.runtime.tag for c in cost_ordered] == ["closed", "half", "open"]


@pytest.mark.unit
def test_cost_policy_preserves_health_filtering() -> None:
    # Arrange — an OPEN-not-eligible candidate with the CHEAPEST hint must still
    # be dropped; price never overrides health. CLOSED ranks before HALF_OPEN.
    open_blocked = _candidate(
        CircuitState.OPEN, recovery_eligible=False, tag="blocked", price_hint=0.01
    )
    half_cheap = _candidate(CircuitState.HALF_OPEN, tag="half", price_hint=1.0)
    closed_pricey = _candidate(CircuitState.CLOSED, tag="closed", price_hint=100.0)

    # Act
    ordered = CostPolicy().order([open_blocked, half_cheap, closed_pricey], _req())

    # Assert — blocked dropped; health tier dominates price
    tags = [c.runtime.tag for c in ordered]
    assert tags == ["closed", "half"]
    assert "blocked" not in tags


@pytest.mark.unit
def test_cost_policy_health_tier_dominates_price() -> None:
    # Arrange — a cheap HALF_OPEN must still sort below a pricey CLOSED.
    closed_pricey = _candidate(CircuitState.CLOSED, tag="closed_pricey", price_hint=50.0)
    half_cheap = _candidate(CircuitState.HALF_OPEN, tag="half_cheap", price_hint=1.0)

    # Act
    ordered = CostPolicy().order([half_cheap, closed_pricey], _req())

    # Assert — health tier dominates the price key
    assert [c.runtime.tag for c in ordered] == ["closed_pricey", "half_cheap"]


@pytest.mark.unit
def test_cost_policy_ordering_is_invariant_under_positive_scaling() -> None:
    # The SDK must do NO price arithmetic — CostPolicy reads the server hint and
    # orders by RELATIVE comparison only. A genuine proof of "relative-only" is
    # SCALE INVARIANCE: multiplying every hint by the same positive constant must
    # not change the order. Any implementation that compared against an absolute
    # threshold, or combined the hint with token counts/rates, would shift under
    # scaling and fail this property.
    def _ordering(scale: float) -> list[str]:
        cands = [
            _candidate(CircuitState.CLOSED, tag="pricey", price_hint=9.0 * scale),
            _candidate(CircuitState.CLOSED, tag="cheap", price_hint=1.0 * scale),
            _candidate(CircuitState.CLOSED, tag="mid", price_hint=4.0 * scale),
        ]
        return [c.runtime.tag for c in CostPolicy().order(cands, _req())]

    baseline = _ordering(1.0)
    # Assert — the relative order is fixed regardless of the absolute magnitude.
    assert baseline == ["cheap", "mid", "pricey"]
    assert _ordering(0.001) == baseline  # scaled down by 1000x
    assert _ordering(1000.0) == baseline  # scaled up by 1000x
    assert _ordering(7.3) == baseline  # arbitrary non-round factor


@pytest.mark.unit
def test_cost_policy_ranks_strictly_by_relative_hint_order() -> None:
    # Reinforce relative-only: the ONLY thing that determines order is which hint
    # is smaller, never the numeric gap or absolute size. Two hints that are very
    # close and two that are far apart order the same way (smaller-first), so no
    # magnitude- or distance-based math is involved.
    close = CostPolicy().order(
        [
            _candidate(CircuitState.CLOSED, tag="hi", price_hint=2.0001),
            _candidate(CircuitState.CLOSED, tag="lo", price_hint=2.0),
        ],
        _req(),
    )
    far = CostPolicy().order(
        [
            _candidate(CircuitState.CLOSED, tag="hi", price_hint=900.0),
            _candidate(CircuitState.CLOSED, tag="lo", price_hint=2.0),
        ],
        _req(),
    )

    # Assert — smaller hint first in BOTH cases; the magnitude of the gap is irrelevant.
    assert [c.runtime.tag for c in close] == ["lo", "hi"]
    assert [c.runtime.tag for c in far] == ["lo", "hi"]


@pytest.mark.unit
def test_cost_policy_is_pure_does_not_touch_breaker() -> None:
    # Arrange
    breaker = MagicMock()
    breaker.admit.side_effect = AssertionError("order() must not call admit()")
    runtime = types.SimpleNamespace(breaker=breaker)
    candidate = ProviderCandidate(
        runtime=runtime,  # type: ignore[arg-type]
        breaker_state=CircuitState.CLOSED,
        recovery_eligible=True,
        translatable=True,
        price_hint=3.0,
    )

    # Act
    CostPolicy().order([candidate], _req())

    # Assert — purity
    breaker.admit.assert_not_called()
    assert breaker.method_calls == []


# ── fix [E]: SelectionPolicy classes are exported from the package root ───
@pytest.mark.unit
def test_selection_policies_importable_from_package_root() -> None:
    # The headline user feature is the constructor selection_policy= arg, so the
    # policy classes must be importable from `solwyn`, not only solwyn._routing.
    import solwyn
    from solwyn import CostPolicy as RootCost
    from solwyn import HealthBasedPolicy as RootHealth
    from solwyn import LatencyPolicy as RootLatency
    from solwyn import SelectionPolicy as RootSelection

    assert RootHealth is HealthBasedPolicy
    assert RootLatency is LatencyPolicy
    assert RootCost is CostPolicy
    # SelectionPolicy is the protocol seam users type their custom policy against.
    from solwyn._routing import SelectionPolicy as PrivateSelection

    assert RootSelection is PrivateSelection

    for name in ("HealthBasedPolicy", "LatencyPolicy", "CostPolicy", "SelectionPolicy"):
        assert name in solwyn.__all__, f"{name} missing from solwyn.__all__"
