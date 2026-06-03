"""Unit tests for solwyn._routing (pure, sans-I/O selection policy).

HealthBasedPolicy.order is a PURE ordering over non-mutating breaker-state
snapshots: it filters out OPEN-not-recovery-eligible candidates and ranks the
rest CLOSED < HALF_OPEN < recovery-eligible OPEN, sinking untranslatable
targets within a state tier. sorted() is stable, so configured (input) order is
preserved inside a priority tier. order() must NEVER call can_proceed() or
mutate a breaker (§4.2 inspection-vs-consumption rule).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from solwyn._routing import (
    HealthBasedPolicy,
    ProviderCandidate,
    RoutingRequest,
)
from solwyn._types import CircuitState, ProviderName


def _candidate(
    state: CircuitState,
    *,
    recovery_eligible: bool = False,
    translatable: bool = True,
    tag: str = "",
) -> ProviderCandidate:
    """Build a ProviderCandidate with a dummy (non-runtime) object in the runtime slot."""
    runtime = types.SimpleNamespace(tag=tag)
    return ProviderCandidate(
        runtime=runtime,  # type: ignore[arg-type]
        breaker_state=state,
        recovery_eligible=recovery_eligible,
        translatable=translatable,
    )


def _req() -> RoutingRequest:
    return RoutingRequest(requested_provider=ProviderName.OPENAI)


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
    breaker.can_proceed.side_effect = AssertionError("order() must not call can_proceed()")

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
    breaker.can_proceed.assert_not_called()
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
    # Assert — extra="forbid" is preserved on the wire-adjacent model
    with pytest.raises(ValueError):
        RoutingRequest(requested_provider=ProviderName.OPENAI, bogus=1)  # type: ignore[call-arg]


@pytest.mark.unit
def test_routing_request_estimated_input_tokens_defaults_zero() -> None:
    # Assert
    assert RoutingRequest(requested_provider=ProviderName.ANTHROPIC).estimated_input_tokens == 0
