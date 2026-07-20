"""Shared test fixtures for Solwyn SDK tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from solwyn._types import ProviderEntry, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.config import SolwynConfig

# Valid credentials that pass format validation
VALID_API_KEY = "sk_proj_" + "a" * 64
VALID_PROJECT_ID = "proj_" + "a" * 24

# A minimal one-link provider chain, enough to satisfy SolwynConfig's
# required ``providers`` invariant in tests that don't care about routing.
DEFAULT_PROVIDER_CHAIN = [ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")]


def _accepted_response(body: dict[str, Any]) -> MagicMock:
    """A 202 httpx.Response stand-in carrying a per-event-disposition body."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 202
    resp.raise_for_status = MagicMock()
    # httpx.Response.json() is sync even on the async client.
    resp.json = MagicMock(return_value=body)
    return resp


def make_mock_client(module: str = "openai._client", name: str = "OpenAI") -> MagicMock:
    """Return a MagicMock that adapter-detection treats as a provider SDK.

    ``with_options(...)`` returns the same mock so per-hop
    ``with_options(timeout=..., max_retries=...)`` resolves back to this
    configured client (the dispatch path calls it before .create()).
    """
    client = MagicMock()
    client.__class__.__module__ = module
    client.__class__.__name__ = name
    client.with_options.return_value = client
    return client


# Standard allow response for budget mock patching
ALLOW_BUDGET_RESPONSE = {
    "allowed": True,
    "remaining_budget": 80.0,
    "reservation_id": "res_123",
    "mode": "alert_only",
    "budget_limit": 100.0,
    "current_usage": 20.0,
    "denied_by_period": None,
    "project_id": VALID_PROJECT_ID,
    "price_hints": None,
}


@pytest.fixture
def mock_httpx_client():
    """Return a mocked httpx.Client."""
    return MagicMock(spec=httpx.Client)


@pytest.fixture
def mock_async_httpx_client():
    """Return a mocked httpx.AsyncClient."""
    return MagicMock(spec=httpx.AsyncClient)


@pytest.fixture
def solwyn_config():
    """Return a SolwynConfig with test defaults.

    SolwynConfig now requires a non-empty ``providers`` chain, so supply a
    single OpenAI entry.
    """
    return SolwynConfig(
        api_key=VALID_API_KEY,
        providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
    )


@pytest.fixture
def circuit_breaker():
    """Return a CircuitBreaker with default thresholds."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=60,
        success_threshold=2,
    )
