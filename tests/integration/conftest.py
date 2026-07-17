"""Shared fixtures for integration tests.

Bootstraps test credentials by signing up a throwaway user and creating
a project via the Solwyn API.  Falls back to env vars when pre-provisioned
credentials are available (CI).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import httpx
import pytest

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter


@dataclass(frozen=True)
class Credentials:
    """Credentials bootstrapped for an integration test session."""

    api_url: str
    api_key: str


@pytest.fixture(scope="session")
def api_url() -> str:
    """Solwyn API base URL."""
    return os.environ.get("SOLWYN_TEST_API_URL", "http://127.0.0.1:8080")


@pytest.fixture(scope="session", autouse=True)
def require_api(api_url: str) -> None:
    """Skip the entire session if the Solwyn API is unreachable."""
    try:
        r = httpx.get(f"{api_url}/health", timeout=3)
        r.raise_for_status()
    except (httpx.HTTPError, OSError):
        pytest.skip("Solwyn API not available")


def _signup_token(http: httpx.Client, session_id: str) -> str:
    """Sign up a throwaway user and return an access token.

    Requires the API's dev signup fast path (Loops disabled — in ../core run
    ``make smoke-api-dev``). A verify-first API (Loops enabled) returns a
    pending-verification challenge whose OTP is emailed and stored hashed, so
    bootstrap is impossible; skip with guidance instead of KeyError.
    """
    email = f"sdk-test-{session_id}@example.com"
    password = f"TestPass!{session_id}"
    r = http.post(
        "/api/v1/auth/signup",
        json={
            "email": email,
            "password": password,
            # The Cloud API's signup requires a name field.
            "display_name": f"SDK Test {session_id}",
        },
    )
    if r.status_code == 409:
        r = http.post("/api/v1/auth/login", json={"email": email, "password": password})
    r.raise_for_status()
    data = r.json()
    if "access_token" not in data:
        pytest.skip(
            "Solwyn API requires email verification for signup; start it with "
            "Loops disabled (in ../core: make smoke-api-dev) or set "
            "SOLWYN_TEST_API_KEY"
        )
    return data["access_token"]


def _create_project(
    http: httpx.Client,
    token: str,
    *,
    name: str,
    budget_limit: float,
    budget_mode: str,
) -> str:
    """Create a project via the API and return its auto-generated API key."""
    r = http.post(
        "/api/v1/projects",
        json={
            "name": name,
            "budget_limit": budget_limit,
            "budget_period": "monthly",
            "budget_mode": budget_mode,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    r.raise_for_status()
    return r.json()["key"]


def _bootstrap_credentials(api_url: str) -> Credentials:
    """Sign up a throwaway user, create an alert-only project, return credentials."""
    session_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        key = _create_project(
            http,
            token,
            name=f"sdk-integ-{session_id}",
            budget_limit=100.0,
            budget_mode="alert_only",
        )
    return Credentials(api_url=api_url, api_key=key)


@pytest.fixture(scope="session")
def test_credentials(api_url: str) -> Credentials:
    """Session-scoped test credentials.

    Uses SOLWYN_TEST_API_KEY if set, otherwise bootstraps via the API.
    """
    env_key = os.environ.get("SOLWYN_TEST_API_KEY")

    if env_key:
        return Credentials(
            api_url=api_url,
            api_key=env_key,
        )

    return _bootstrap_credentials(api_url)


# ---------------------------------------------------------------------------
# SDK component fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_enforcer(test_credentials: Credentials) -> BudgetEnforcer:
    """Sync BudgetEnforcer pointed at the live API."""
    enforcer = BudgetEnforcer(
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        budget_mode=BudgetMode.ALERT_ONLY,
        fail_open=True,
    )
    yield enforcer
    enforcer.close()


@pytest.fixture
async def async_budget_enforcer(test_credentials: Credentials) -> AsyncBudgetEnforcer:
    """Async BudgetEnforcer pointed at the live API."""
    enforcer = AsyncBudgetEnforcer(
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        budget_mode=BudgetMode.ALERT_ONLY,
        fail_open=True,
    )
    yield enforcer
    await enforcer.close()


@pytest.fixture
def metadata_reporter(test_credentials: Credentials) -> MetadataReporter:
    """Sync MetadataReporter pointed at the live API."""
    reporter = MetadataReporter(
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        flush_interval=1.0,
    )
    yield reporter
    reporter.close()


@pytest.fixture
async def async_metadata_reporter(test_credentials: Credentials) -> AsyncMetadataReporter:
    """Async MetadataReporter pointed at the live API."""
    reporter = AsyncMetadataReporter(
        api_url=test_credentials.api_url,
        api_key=test_credentials.api_key,
        flush_interval=1.0,
    )
    await reporter.start()
    yield reporter
    await reporter.close()


@pytest.fixture(scope="session")
def hard_denied_credentials(api_url: str) -> Credentials:
    """Credentials for a project driven OVER its hard_deny budget limit.

    Bootstraps a dedicated project (hard_deny, $0.05 limit) and burns it with
    one large confirmed spend (~$2.50 of gpt-4o tokens), then verifies with a
    fresh enforcer (no allow-cache) that the API now denies. E2E tests use
    these credentials to prove BudgetExceededError surfaces from the wrapper
    WITHOUT the provider being called. Session-scoped: usage persists.

    Always bootstraps its own project — SOLWYN_TEST_API_KEY cannot be used
    (that project must stay usable for happy-path tests).
    """
    session_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        key = _create_project(
            http,
            token,
            name=f"sdk-deny-{session_id}",
            budget_limit=0.05,
            budget_mode="hard_deny",
        )

    burner = BudgetEnforcer(
        api_url=api_url, api_key=key, budget_mode=BudgetMode.HARD_DENY, fail_open=False
    )
    try:
        check = burner.check_budget(estimated_input_tokens=100, model="gpt-4o", provider="openai")
        if check.reservation_id is None:
            pytest.fail("hard-deny bootstrap: expected a reservation on a fresh project")
        burner.confirm_cost(
            reservation_id=check.reservation_id,
            model="gpt-4o",
            token_details=TokenDetails(input_tokens=200_000, output_tokens=200_000),
            provider="openai",
            call_id=f"harness-burn-{session_id}",
        )
    finally:
        burner.close()

    # Fresh enforcer: its allow-cache is empty, so this check hits the API.
    verifier = BudgetEnforcer(
        api_url=api_url, api_key=key, budget_mode=BudgetMode.HARD_DENY, fail_open=False
    )
    try:
        denied = verifier.check_budget(
            estimated_input_tokens=100, model="gpt-4o", provider="openai"
        )
    finally:
        verifier.close()
    if denied.allowed:
        pytest.fail(
            "hard-deny bootstrap: burn did not trip the budget "
            f"(usage=${denied.current_usage:.2f}, limit=${denied.budget_limit:.2f})"
        )
    return Credentials(api_url=api_url, api_key=key)


# Reusable token details for confirm calls
SAMPLE_TOKEN_DETAILS = TokenDetails(input_tokens=100, output_tokens=50)
