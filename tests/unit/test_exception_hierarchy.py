"""SDK exceptions must share a base class and be importable from the package root."""

from __future__ import annotations

import pytest

import solwyn
from solwyn import (
    BudgetExceededError,
    ConfigurationError,
    ProviderUnavailableError,
    RunStoppedError,
    SolwynError,
    UntranslatableModelError,
    UntranslatableRequestError,
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
def test_run_stopped_error_is_public_and_preserves_budget_compatibility() -> None:
    assert RunStoppedError is solwyn.exceptions.RunStoppedError
    assert issubclass(RunStoppedError, BudgetExceededError)

    exc = RunStoppedError(
        agent_run_id="run_abc",
        project_id="proj_" + "a" * 24,
        budget_limit=100.0,
        current_usage=25.0,
        estimated_cost=1.5,
        mode="hard_deny",
    )

    assert isinstance(exc, BudgetExceededError)
    assert str(exc) == "Run run_abc was stopped from the Solwyn dashboard"
    assert exc.agent_run_id == "run_abc"
    assert exc.project_id == "proj_" + "a" * 24
    assert exc.budget_limit == 100.0
    assert exc.current_usage == 25.0
    assert exc.estimated_cost == 1.5
    assert exc.budget_period == "run_stopped"
    assert exc.mode == "hard_deny"


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
class TestUntranslatableRequestError:
    """Fail-loud cross-provider translation error — STRUCTURAL labels only, never content."""

    def test_is_solwyn_error(self) -> None:
        exc = UntranslatableRequestError(
            source="openai",
            target="anthropic",
            feature="response_format",
        )
        assert isinstance(exc, SolwynError)
        assert issubclass(UntranslatableRequestError, SolwynError)

    def test_structural_fields_stored(self) -> None:
        exc = UntranslatableRequestError(
            source="openai",
            target="anthropic",
            feature="temperature>1.0",
        )
        assert exc.source == "openai"
        assert exc.target == "anthropic"
        assert exc.feature == "temperature>1.0"

    def test_message_carries_structural_labels(self) -> None:
        exc = UntranslatableRequestError(
            source="anthropic",
            target="openai",
            feature="anthropic.computer_use",
        )
        message = str(exc)
        assert "anthropic.computer_use" in message
        assert "anthropic" in message
        assert "openai" in message

    def test_constructor_is_keyword_only_structural(self) -> None:
        # Constructing requires ONLY structural labels; positional/content args rejected.
        with pytest.raises(TypeError):
            UntranslatableRequestError("openai", "anthropic", "response_format")  # type: ignore[misc]

    def test_constructor_rejects_unknown_content_kwargs(self) -> None:
        # No place to smuggle prompt content (e.g. a `value`/`content` kwarg).
        with pytest.raises(TypeError):
            UntranslatableRequestError(  # type: ignore[call-arg]
                source="openai",
                target="anthropic",
                feature="response_format",
                value="customer prompt text",
            )


@pytest.mark.unit
class TestUntranslatableModelError:
    """Raised when a model id is not configured for the target provider (config, not content)."""

    def test_is_solwyn_error(self) -> None:
        exc = UntranslatableModelError(model="gpt-5.5", provider="anthropic")
        assert isinstance(exc, SolwynError)
        assert issubclass(UntranslatableModelError, SolwynError)

    def test_structural_fields_stored(self) -> None:
        exc = UntranslatableModelError(model="gpt-5.5", provider="anthropic")
        assert exc.model == "gpt-5.5"
        assert exc.provider == "anthropic"

    def test_message_carries_model_and_provider(self) -> None:
        exc = UntranslatableModelError(model="claude-sonnet-5", provider="openai")
        message = str(exc)
        assert "claude-sonnet-5" in message
        assert "openai" in message

    def test_constructor_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            UntranslatableModelError("gpt-5.5", "anthropic")  # type: ignore[misc]


@pytest.mark.unit
def test_budget_exceeded_error_repr_remains_compatible() -> None:
    exc = BudgetExceededError(
        project_id="proj_" + "a" * 24,
        budget_limit=100.0,
        current_usage=120.0,
        estimated_cost=5.0,
        budget_period="daily",
        mode="hard_deny",
    )

    assert repr(exc) == "BudgetExceededError(budget_limit=100.0, current_usage=120.0)"


@pytest.mark.unit
def test_run_stopped_error_repr_names_its_public_type() -> None:
    exc = RunStoppedError(
        agent_run_id="run_abc",
        project_id="proj_" + "a" * 24,
        budget_limit=100.0,
        current_usage=120.0,
        estimated_cost=5.0,
        mode="hard_deny",
    )

    assert repr(exc) == "RunStoppedError(budget_limit=100.0, current_usage=120.0)"
