"""SDK exceptions must share a base class and be importable from the package root."""

from __future__ import annotations

import pytest

from solwyn import (
    BudgetExceededError,
    ConfigurationError,
    ProviderUnavailableError,
    SolwynError,
)


@pytest.mark.unit
def test_solwyn_error_is_importable_from_package_root() -> None:
    import solwyn

    assert hasattr(solwyn, "SolwynError"), "solwyn.SolwynError must be exported"


@pytest.mark.unit
def test_all_sdk_exceptions_inherit_from_solwyn_error() -> None:
    assert issubclass(BudgetExceededError, SolwynError)
    assert issubclass(ProviderUnavailableError, SolwynError)
    assert issubclass(ConfigurationError, SolwynError)


@pytest.mark.unit
def test_solwyn_error_catches_all_families() -> None:
    cases: list[tuple[type[SolwynError], dict[str, object]]] = [
        (
            BudgetExceededError,
            {
                "project_id": "proj_" + "a" * 24,
                "budget_limit": 100.0,
                "current_usage": 120.0,
                "estimated_cost": 5.0,
                "budget_period": "daily",
                "mode": "hard_deny",
            },
        ),
        (ProviderUnavailableError, {"provider": "openai", "circuit_state": "open"}),
        (ConfigurationError, {"field": "api_key"}),
    ]
    for exc_class, kwargs in cases:
        try:
            if exc_class is BudgetExceededError:
                raise exc_class(**kwargs)  # type: ignore[arg-type]
            raise exc_class("test", **kwargs)  # type: ignore[arg-type]
        except SolwynError:
            pass  # expected
        except Exception as err:
            raise AssertionError(f"{exc_class.__name__} did not match SolwynError") from err


@pytest.mark.unit
class TestProviderUnavailableErrorChain:
    """ProviderUnavailableError carries the attempted chain (new dispatch)."""

    def test_legacy_construction_still_works(self) -> None:
        # Existing call sites pass provider + circuit_state.
        exc = ProviderUnavailableError(
            "openai circuit open",
            provider="openai",
            circuit_state="open",
        )
        assert exc.provider == "openai"
        assert exc.circuit_state == "open"
        assert exc.attempted is None

    def test_attempted_chain_construction(self) -> None:
        # New dispatch surfaces the exhausted chain.
        exc = ProviderUnavailableError(
            "all providers exhausted",
            attempted=["openai", "anthropic"],
        )
        assert exc.attempted == ["openai", "anthropic"]
        # provider/circuit_state are optional now.
        assert exc.provider is None
        assert exc.circuit_state is None

    def test_message_only_construction(self) -> None:
        exc = ProviderUnavailableError("nothing available")
        assert exc.provider is None
        assert exc.circuit_state is None
        assert exc.attempted is None

    def test_is_solwyn_error(self) -> None:
        exc = ProviderUnavailableError("x", attempted=["openai"])
        assert isinstance(exc, SolwynError)


@pytest.mark.unit
def test_exceptions_have_useful_repr() -> None:
    exc = BudgetExceededError(
        project_id="proj_" + "a" * 24,
        budget_limit=100.0,
        current_usage=120.0,
        estimated_cost=5.0,
        budget_period="daily",
        mode="hard_deny",
    )
    rep = repr(exc)
    assert "BudgetExceededError" in rep
    assert "100" in rep
