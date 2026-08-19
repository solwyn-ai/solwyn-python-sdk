"""Single-credential constructor tests for the tier-zero project key API."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from solwyn import Solwyn
from solwyn.exceptions import ConfigurationError

VALID_PROJECT_KEY = "sk_proj_" + "a" * 64


@pytest.fixture(autouse=True)
def _clean_solwyn_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove host SOLWYN_* env vars so tests control config completely."""
    for key in list(os.environ):
        if key.startswith("SOLWYN_"):
            monkeypatch.delenv(key)


@pytest.fixture
def openai_client() -> MagicMock:
    """Create a mock that provider detection treats as openai.OpenAI()."""
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    return client


def _make_solwyn(client: object, **config_kwargs: object) -> Solwyn:
    """Create a Solwyn wrapper without running the reporter background loop."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **config_kwargs)
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    return solwyn


@pytest.mark.unit
def test_solwyn_constructor_accepts_only_api_key(openai_client: MagicMock) -> None:
    """Tier-zero: project_id is no longer a constructor parameter."""
    client = _make_solwyn(openai_client, api_key=VALID_PROJECT_KEY)

    assert client._solwyn_config.api_key == VALID_PROJECT_KEY
    assert not hasattr(client._solwyn_config, "project_id")

    client.close()


@pytest.mark.unit
def test_solwyn_rejects_legacy_project_id_kwarg(openai_client: MagicMock) -> None:
    """Passing project_id should fail clearly."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'project_id'"):
        _make_solwyn(
            openai_client,
            api_key=VALID_PROJECT_KEY,
            project_id="proj_" + "0" * 24,
        )


@pytest.mark.unit
def test_solwyn_loads_only_solwyn_api_key_from_env(
    openai_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SOLWYN_PROJECT_ID is no longer recognized."""
    monkeypatch.setenv("SOLWYN_API_KEY", VALID_PROJECT_KEY)
    monkeypatch.setenv("SOLWYN_PROJECT_ID", "proj_" + "0" * 24)

    client = _make_solwyn(openai_client)

    assert client._solwyn_config.api_key == VALID_PROJECT_KEY
    assert not hasattr(client._solwyn_config, "project_id")

    client.close()


@pytest.mark.unit
def test_solwyn_rejects_legacy_sk_solwyn_prefix(openai_client: MagicMock) -> None:
    """Legacy sk_solwyn_* keys are rejected after the project-key rename."""
    with pytest.raises(ConfigurationError, match="must start with 'sk_proj_'"):
        _make_solwyn(openai_client, api_key="sk_solwyn_" + "a" * 64)
