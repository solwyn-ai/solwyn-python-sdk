"""Shared test fixtures for Solwyn SDK tests."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
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

# Namespace for call_uuid below. Arbitrary but FIXED: the ids it derives are
# stable across runs, so a queue-order assertion can name them.
_CALL_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_MISSING = object()


@contextmanager
def patch_wrapper_local(wrapper: object, name: str, replacement: Any) -> Iterator[Any]:
    """Temporarily install a test double directly on a Solwyn wrapper."""
    previous = vars(wrapper).get(name, _MISSING)
    object.__setattr__(wrapper, name, replacement)
    try:
        yield replacement
    finally:
        if previous is _MISSING:
            vars(wrapper).pop(name, None)
        else:
            object.__setattr__(wrapper, name, previous)


@pytest.fixture
def repo_tool_env() -> Callable[[Path], dict[str, str]]:
    """Build an isolated subprocess environment for checkout-owned tools."""

    def build(checkout: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str((checkout / "src").resolve())
        return environment

    return build


def call_uuid(label: str) -> str:
    """A canonical-UUID call_id derived from a readable *label*.

    ``BudgetConfirmRequest.call_id`` is pinned to the canonical lowercase UUID
    text form (the wire contract — see ``_constants.CALL_ID_PATTERN``), so a
    test cannot say ``call_id="a"`` any more. Deriving the id from the label
    keeps the intent legible at both ends: ``call_uuid("a")`` in the arrangement
    and ``call_uuid("a")`` in the assertion are the same id, and it is stable
    across runs and processes.
    """
    return str(uuid.uuid5(_CALL_ID_NAMESPACE, label))


# Loggers written by the reporter's BACKGROUND delivery machinery (flush
# threads, exit hooks). caplog installs its handler on the ROOT logger, so these
# leak into any test's capture window — and since delivery is at-least-once, a
# reporter left live by an earlier test retries against the unreachable test API
# for several backoff cycles, emitting WARNINGs the whole time. Tests asserting
# "the logger I named stayed silent" must not be tripped by that noise.
_BACKGROUND_LOGGERS = ("solwyn.reporter", "solwyn._lifecycle")


def foreground_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Captured records excluding the background delivery loggers.

    Use instead of ``caplog.records`` when asserting that the logger named in
    ``caplog.at_level(..., logger=...)`` emitted nothing.
    """
    return [r for r in caplog.records if not r.name.startswith(_BACKGROUND_LOGGERS)]


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
