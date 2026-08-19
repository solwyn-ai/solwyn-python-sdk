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


@pytest.mark.unit
def test_untracked_spend_surface_error_is_publicly_exported() -> None:
    assert "UntrackedSpendSurfaceError" in solwyn.__all__
    assert solwyn.UntrackedSpendSurfaceError is exceptions.UntrackedSpendSurfaceError
    assert issubclass(solwyn.UntrackedSpendSurfaceError, solwyn.SolwynError)
    assert issubclass(solwyn.UntrackedSpendSurfaceError, AttributeError)


@pytest.mark.unit
def test_run_stopped_error_is_publicly_exported() -> None:
    assert "RunStoppedError" in solwyn.__all__
    assert solwyn.RunStoppedError is exceptions.RunStoppedError
    assert issubclass(solwyn.RunStoppedError, solwyn.SolwynError)
    assert not issubclass(solwyn.RunStoppedError, solwyn.BudgetExceededError)


@pytest.mark.unit
def test_tags_clamped_warning_is_publicly_exported() -> None:
    assert "SolwynTagsClampedWarning" in solwyn.__all__
    assert solwyn.SolwynTagsClampedWarning is exceptions.SolwynTagsClampedWarning
    assert issubclass(solwyn.SolwynTagsClampedWarning, UserWarning)


@pytest.mark.unit
def test_run_context_and_tag_bounds_are_publicly_exported() -> None:
    expected = {
        "current_run_context",
        "RunContext",
        "TAGS_MAX_KEYS",
        "TAG_KEY_MAX_LENGTH",
        "TAG_VALUE_MAX_LENGTH",
    }

    assert expected <= set(solwyn.__all__)
    assert solwyn.TAGS_MAX_KEYS == 10
    assert solwyn.TAG_KEY_MAX_LENGTH == 64
    assert solwyn.TAG_VALUE_MAX_LENGTH == 256


@pytest.mark.unit
def test_run_handle_api_is_publicly_exported() -> None:
    expected = {"RunHandle", "create_run", "start_run"}

    assert expected <= set(solwyn.__all__)
    assert solwyn.RunHandle is not None
    assert callable(solwyn.create_run)
    assert callable(solwyn.start_run)
