"""Tests for env-var config loading and credential validation via constructors."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._types import BudgetMode, ProviderEntry, ProviderName
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.config import SolwynConfig
from solwyn.exceptions import ConfigurationError, SolwynError


@pytest.fixture(autouse=True)
def _clean_solwyn_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all SOLWYN_* env vars to prevent test pollution from host environment."""
    for key in list(os.environ):
        if key.startswith("SOLWYN_"):
            monkeypatch.delenv(key)


def _mock_openai_client() -> MagicMock:
    """Create a mock that looks like openai.OpenAI() for provider detection."""
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    return client


def _make_solwyn(client: object, **config_kwargs: object) -> Solwyn:
    """Create a Solwyn wrapper with mocked reporter thread."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **config_kwargs)
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    return solwyn


@pytest.mark.unit
class TestEnvVarConstruction:
    """Solwyn(client) loads config from SOLWYN_* env vars when no kwargs given."""

    def test_env_vars_populate_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SOLWYN_API_KEY in env is enough for construction."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.api_key == VALID_API_KEY
        assert not hasattr(solwyn._config, "project_id")

        solwyn.close()

    def test_kwargs_override_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor kwargs take precedence over env vars."""
        env_key = "sk_proj_" + "b" * 64
        monkeypatch.setenv("SOLWYN_API_KEY", env_key)

        client = _mock_openai_client()
        solwyn = _make_solwyn(client, api_key=VALID_API_KEY)

        assert solwyn._config.api_key == VALID_API_KEY
        assert not hasattr(solwyn._config, "project_id")

        solwyn.close()

    def test_missing_api_key_env_var_raises(self) -> None:
        """No SOLWYN_API_KEY in env and no kwarg -> ConfigurationError."""
        client = _mock_openai_client()

        with pytest.raises(ConfigurationError) as exc_info:
            _make_solwyn(client)

        assert exc_info.value.field == "api_key"

    def test_solwyn_project_id_env_var_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SOLWYN_PROJECT_ID is not part of SDK configuration anymore."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_PROJECT_ID", "proj_" + "0" * 24)

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert not hasattr(solwyn._config, "project_id")

        solwyn.close()

    def test_api_url_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SOLWYN_API_URL env var overrides the default api_url."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_API_URL", "https://custom.solwyn.ai")

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.api_url == "https://custom.solwyn.ai"

        solwyn.close()

    def test_budget_mode_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SOLWYN_BUDGET_MODE env var overrides the default budget_mode."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_BUDGET_MODE", "hard_deny")

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.budget_mode == BudgetMode.HARD_DENY

        solwyn.close()

    @pytest.mark.parametrize(
        ("env_val", "expected"),
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_fail_open_boolean_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_val: str,
        expected: bool,
    ) -> None:
        """SOLWYN_FAIL_OPEN coerces string values to booleans."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)
        monkeypatch.setenv("SOLWYN_FAIL_OPEN", env_val)

        client = _mock_openai_client()
        solwyn = _make_solwyn(client)

        assert solwyn._config.fail_open is expected

        solwyn.close()


@pytest.mark.unit
class TestConfigurationErrorFromBadCredentials:
    """Malformed api_key raises ConfigurationError with correct attributes."""

    def test_bad_api_key_prefix_raises_configuration_error(self) -> None:
        """api_key without sk_proj_ prefix -> ConfigurationError(field='api_key')."""
        client = _mock_openai_client()

        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="bad_key")

        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_api_key_too_short_raises_configuration_error(self) -> None:
        """api_key with fewer than 64 chars after prefix -> ConfigurationError."""
        client = _mock_openai_client()

        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="sk_proj_short")

        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_api_key_too_long_raises_configuration_error(self) -> None:
        """api_key with more than 64 chars after prefix -> ConfigurationError."""
        client = _mock_openai_client()

        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="sk_proj_" + "a" * 65)

        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_empty_api_key_raises_configuration_error(self) -> None:
        """Empty string api_key -> ConfigurationError(field='api_key')."""
        client = _mock_openai_client()

        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="")

        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_unicode_homograph_api_key_raises_configuration_error(self) -> None:
        """Unicode homograph in api_key (Cyrillic a) -> ConfigurationError."""
        client = _mock_openai_client()
        bad_key = "sk_proj_" + "\u0430" * 64  # Cyrillic 'а' (U+0430)

        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key=bad_key)

        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_path_traversal_api_key_raises_configuration_error(self) -> None:
        """Path traversal in api_key -> ConfigurationError(field='api_key')."""
        client = _mock_openai_client()

        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(client, api_key="sk_proj_../etc")

        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_configuration_error_catchable_as_solwyn_error(self) -> None:
        """ConfigurationError can be caught as SolwynError."""
        client = _mock_openai_client()

        with pytest.raises(SolwynError):
            Solwyn(client, api_key="bad_key")


def _make_async_solwyn(client: object, **config_kwargs: object) -> AsyncSolwyn:
    """Create an AsyncSolwyn wrapper."""
    return AsyncSolwyn(client, **config_kwargs)


@pytest.mark.unit
class TestAsyncSolwynConstructors:
    """AsyncSolwyn constructor shares the same config path as Solwyn."""

    @pytest.mark.asyncio
    async def test_async_env_vars_populate_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AsyncSolwyn loads SOLWYN_API_KEY from env."""
        monkeypatch.setenv("SOLWYN_API_KEY", VALID_API_KEY)

        client = _mock_openai_client()
        solwyn = _make_async_solwyn(client)

        assert solwyn._config.api_key == VALID_API_KEY
        assert not hasattr(solwyn._config, "project_id")

        await solwyn.close()

    def test_async_bad_api_key_raises_configuration_error(self) -> None:
        """AsyncSolwyn raises ConfigurationError for malformed api_key."""
        client = _mock_openai_client()

        with pytest.raises(ConfigurationError) as exc_info:
            AsyncSolwyn(client, api_key="bad_key")

        assert exc_info.value.field == "api_key"
        assert isinstance(exc_info.value.message, str)
        assert len(exc_info.value.message) > 0

    def test_async_project_id_kwarg_is_rejected(self) -> None:
        """AsyncSolwyn no longer accepts project_id."""
        client = _mock_openai_client()

        with pytest.raises(TypeError, match="unexpected keyword argument 'project_id'"):
            AsyncSolwyn(client, api_key=VALID_API_KEY, project_id="proj_" + "0" * 24)


@pytest.mark.unit
class TestProvidersChainRequired:
    """SolwynConfig requires a non-empty providers chain."""

    def test_empty_chain_raises_configuration_error(self) -> None:
        """_check_chain raises ConfigurationError when providers is empty."""
        with pytest.raises(ConfigurationError) as exc_info:
            SolwynConfig(api_key=VALID_API_KEY, providers=[])

        assert exc_info.value.field == "providers"

    def test_single_entry_chain_is_valid(self) -> None:
        """A chain with one entry round-trips and is the primary."""
        config = SolwynConfig(
            api_key=VALID_API_KEY,
            providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-4o")],
        )
        assert len(config.providers) == 1
        assert config.providers[0].provider == ProviderName.OPENAI
        assert config.providers[0].model == "gpt-4o"

    def test_provider_entry_round_trips(self) -> None:
        """ProviderEntry serializes and reconstructs without loss."""
        entry = ProviderEntry(
            provider=ProviderName.ANTHROPIC,
            model="claude-3-5-sonnet",
            default_params={"max_tokens": 256},
        )
        rebuilt = ProviderEntry.model_validate(entry.model_dump())
        assert rebuilt == entry
        assert rebuilt.default_params == {"max_tokens": 256}


@pytest.mark.unit
class TestFailoverKnobDefaults:
    """Failover tuning knobs carry their default values."""

    def test_failover_knob_defaults(self) -> None:
        config = SolwynConfig(
            api_key=VALID_API_KEY,
            providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-4o")],
        )
        assert config.failover_total_timeout == 30.0
        assert config.failover_idempotency == "safe"
        assert config.same_provider_retries == 0
        assert config.circuit_breaker_recovery_timeout_jitter == 0.2
        assert config.default_params == {}

    def test_failover_knobs_are_overridable(self) -> None:
        config = SolwynConfig(
            api_key=VALID_API_KEY,
            providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-4o")],
            failover_total_timeout=12.5,
            failover_idempotency="always",
            same_provider_retries=2,
            default_params={"temperature": 0.0},
        )
        assert config.failover_total_timeout == 12.5
        assert config.failover_idempotency == "always"
        assert config.same_provider_retries == 2
        assert config.default_params == {"temperature": 0.0}
