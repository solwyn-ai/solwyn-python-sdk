"""Public package exports for failover configuration and observability."""

from __future__ import annotations

import pytest

import solwyn


@pytest.mark.unit
def test_failover_types_are_publicly_exported() -> None:
    expected = {
        "ProviderEntry",
        "ProviderName",
        "FailoverReason",
        "RoutingRequest",
        "ProviderCandidate",
        "CircuitState",
        "CircuitBreakerState",
    }

    assert expected <= set(solwyn.__all__)
    for name in expected:
        assert getattr(solwyn, name) is not None
