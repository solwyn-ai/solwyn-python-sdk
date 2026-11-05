"""Constructor credential tests: one account API key plus the project it names."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from solwyn import Solwyn
from solwyn.exceptions import ConfigurationError

VALID_KEY = "solwyn_sk_" + "a" * 64
PROJECT = "checkout-bot"


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
def test_solwyn_constructor_takes_api_key_and_project(openai_client: MagicMock) -> None:
    client = _make_solwyn(openai_client, api_key=VALID_KEY, project=PROJECT)

    assert client._solwyn_config.api_key == VALID_KEY
    assert client._solwyn_config.project == PROJECT
    assert client._solwyn_binding.selector == PROJECT
    assert client._solwyn_binding.state == "unvalidated"
    assert not hasattr(client._solwyn_config, "project_id")

    client.close()


@pytest.mark.unit
def test_solwyn_rejects_project_id_kwarg(openai_client: MagicMock) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        _make_solwyn(
            openai_client,
            api_key=VALID_KEY,
            project_id="proj_" + "0" * 24,
        )

    assert excinfo.value.field == "project_id"


@pytest.mark.unit
def test_solwyn_loads_api_key_and_project_from_env(
    openai_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOLWYN_API_KEY", VALID_KEY)
    monkeypatch.setenv("SOLWYN_PROJECT", PROJECT)

    client = _make_solwyn(openai_client)

    assert client._solwyn_config.api_key == VALID_KEY
    assert client._solwyn_config.project == PROJECT
    client.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    ["sk_proj_" + "a" * 64, "sk_solwyn_" + "a" * 64],
    ids=["old-project-key", "tier-zero-key"],
)
def test_solwyn_rejects_retired_key_shapes(openai_client: MagicMock, key: str) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        _make_solwyn(openai_client, api_key=key, project=PROJECT)

    assert excinfo.value.field == "api_key"
