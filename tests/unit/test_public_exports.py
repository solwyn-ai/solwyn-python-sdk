"""Public package exports for failover configuration and observability."""

from __future__ import annotations

import pytest

import solwyn
from solwyn import exceptions


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


@pytest.mark.unit
def test_unsupported_surface_error_is_publicly_exported() -> None:
    # Raised by client.py and the provider adapters when an adapter does not
    # serve a requested media surface, so callers must be able to catch it
    # from the package root.
    assert "UnsupportedSurfaceError" in solwyn.__all__
    assert solwyn.UnsupportedSurfaceError is exceptions.UnsupportedSurfaceError
    assert issubclass(solwyn.UnsupportedSurfaceError, solwyn.SolwynError)
