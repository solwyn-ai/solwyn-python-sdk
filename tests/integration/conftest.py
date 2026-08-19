"""Shared fixtures for integration tests.

Bootstraps test credentials by signing up a throwaway user and creating
a project via the Solwyn API.  Falls back to env vars when pre-provisioned
credentials are available (CI).
"""

from __future__ import annotations

import inspect
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fake_provider import (
    PORT_PROFILES,
    FakeAnthropicServer,
    FakeProviderServer,
    start_on_conventional_port,
)

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode, MetadataEvent
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter


@dataclass(frozen=True)
class Credentials:
    """Credentials bootstrapped for an integration test session."""

    api_url: str
    api_key: str


@dataclass(frozen=True)
class ProjectCredentials(Credentials):
    """Credentials plus the project id, for tests that read budget status.

    The lease tests (PJ-2) observe server-side reclaim through
    ``GET /projects/{id}/budget`` — ``outage_overspend_ceiling_usd`` and
    ``current_usage`` — so they need the id the API key resolves to.
    """

    project_id: str


@pytest.fixture(scope="session")
def api_url() -> str:
    """Solwyn API base URL."""
    return os.environ.get("SOLWYN_TEST_API_URL", "http://127.0.0.1:8080")


_REQUIRE_LIVE_API = True


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Let a direct fake-provider-only run proceed without the control plane."""
    global _REQUIRE_LIVE_API
    _REQUIRE_LIVE_API = any("fake_provider_harness_only" not in item.fixturenames for item in items)


@pytest.fixture(scope="session", autouse=True)
def require_api(api_url: str) -> None:
    """Skip live-pipeline runs if the Solwyn API is unreachable."""
    if not _REQUIRE_LIVE_API:
        return
    try:
        r = httpx.get(f"{api_url}/health", timeout=3)
        r.raise_for_status()
    except (httpx.HTTPError, OSError):
        pytest.skip("Solwyn API not available")


@pytest.fixture
def fake_provider_harness_only() -> None:
    """Mark a direct fake-provider test that has no Solwyn API dependency."""


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
) -> dict[str, Any]:
    """Create a project via the API and return the created project body."""
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
    body = r.json()
    if not isinstance(body, dict):
        pytest.fail(f"project create returned non-object JSON: {body!r}")
    return body


def provision_project(
    api_url: str, *, name: str, budget_limit: float, budget_mode: str
) -> ProjectCredentials:
    """Sign up a throwaway account and create ONE project on it.

    One account per project, deliberately: the free tier caps how many projects
    an account may hold, so every test that needs its own budget posture brings
    its own account — the pattern ``hard_denied_credentials`` and
    ``run_capped_credentials`` already follow.
    """
    session_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        project = _create_project(
            http,
            token,
            name=f"{name}-{session_id}",
            budget_limit=budget_limit,
            budget_mode=budget_mode,
        )
    return ProjectCredentials(api_url=api_url, api_key=project["key"], project_id=project["id"])


def burn_budget(
    credentials: Credentials,
    *,
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-5.5",
) -> None:
    """Drive real server-side spend: one legacy check + its confirm.

    Used to place a project on the far side of its cap before the lease path is
    exercised. Fails loudly rather than silently under-burning — a denied check
    hands back no reservation, so there would be nothing to confirm.
    """
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        headers = {"Authorization": f"Bearer {credentials.api_key}"}
        r = http.post(
            "/api/v1/budgets/check",
            json={"estimated_input_tokens": 100, "model": model, "provider": "openai"},
            headers=headers,
        )
        r.raise_for_status()
        reservation_id = r.json().get("reservation_id")
        if not isinstance(reservation_id, str):
            pytest.fail(f"burn_budget: no reservation to confirm against: {r.json()}")
        r = http.post(
            "/api/v1/budgets/confirm",
            json={
                "reservation_id": reservation_id,
                "model": model,
                "provider": "openai",
                "call_id": str(uuid.uuid4()),
                "token_details": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            },
            headers=headers,
        )
        r.raise_for_status()


def budget_status(credentials: ProjectCredentials) -> dict[str, Any]:
    """The project's live budget status (usage + the lease overspend ceiling)."""
    with httpx.Client(base_url=credentials.api_url, timeout=10) as http:
        r = http.get(
            f"/api/v1/projects/{credentials.project_id}/budget",
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        r.raise_for_status()
        body = r.json()
    if not isinstance(body, dict):
        pytest.fail(f"budget status returned non-object JSON: {body!r}")
    return body


def _bootstrap_credentials(api_url: str) -> Credentials:
    """Sign up a throwaway user, create an alert-only project, return credentials."""
    session_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        project = _create_project(
            http,
            token,
            name=f"sdk-integ-{session_id}",
            budget_limit=100.0,
            budget_mode="alert_only",
        )
    return Credentials(api_url=api_url, api_key=project["key"])


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
    reporter.start()
    yield reporter
    await reporter.close()


@pytest.fixture(scope="session")
def hard_denied_credentials(api_url: str) -> Credentials:
    """Credentials for a project driven OVER its hard_deny budget limit.

    Bootstraps a dedicated project (hard_deny, $0.05 limit) and burns it with
    one large confirmed spend (~$2.50 of gpt-5.5 tokens), then verifies with a
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
        )["key"]

    burner = BudgetEnforcer(
        api_url=api_url, api_key=key, budget_mode=BudgetMode.HARD_DENY, fail_open=False
    )
    try:
        check = burner.check_budget(estimated_input_tokens=100, model="gpt-5.5", provider="openai")
        if check.reservation_id is None:
            pytest.fail("hard-deny bootstrap: expected a reservation on a fresh project")
        # Settlement lives in the reporter now (no enforcer.confirm_cost): build
        # the confirm sans-I/O on the enforcer, then send it synchronously via a
        # throwaway reporter's _send_confirm so the burn lands before we verify.
        confirm = burner.build_confirm_request(
            reservation_id=check.reservation_id,
            model="gpt-5.5",
            token_details=TokenDetails(input_tokens=200_000, output_tokens=200_000),
            provider="openai",
            # The API pins call_id to the uuid4 shape the SDK's client emits
            # (shared CALL_ID_PATTERN); a descriptive label is a 422.
            call_id=str(uuid.uuid4()),
        )
        burn_reporter = MetadataReporter(api_url=api_url, api_key=key)
        try:
            burn_reporter._send_confirm(confirm)
        finally:
            burn_reporter.close()
    finally:
        burner.close()

    # Fresh enforcer: its allow-cache is empty, so this check hits the API.
    verifier = BudgetEnforcer(
        api_url=api_url, api_key=key, budget_mode=BudgetMode.HARD_DENY, fail_open=False
    )
    try:
        denied = verifier.check_budget(
            estimated_input_tokens=100, model="gpt-5.5", provider="openai"
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


# ---------------------------------------------------------------------------
# E2E wrapper harness
#
# These fixtures build REAL Solwyn(OpenAI(...)) wrappers against the local
# fake provider server (fake_provider.py) and the live Solwyn API — the only
# place the full interception pipeline (client.py, _proxies.py, stream.py,
# providers/) runs against real HTTP on both sides.
#
# Conventions the tests rely on:
# - budget_check_cache_ttl=0 on every wrapped client: the enforcer's allow-
#   cache returns reservation_id=None on hits, which silently skips confirm
#   for a second same-client call inside the 5s default TTL.
# - Happy-path calls use a model the API prices (gpt-5.5): unpriced models are
#   allowed WITHOUT a reservation, so confirm never fires.
# - WireRecorder wraps private seams (client._solwyn_budget / client._solwyn_reporter) by
#   design: it records the wire-bound payloads while REAL delivery to the live
#   API still happens (precedent: test_metadata_ingest asserts on _queue).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fake_provider() -> Iterator[FakeProviderServer]:
    """Ephemeral-port fake provider — detected as the generic catch-all."""
    with FakeProviderServer() as server:
        yield server


@pytest.fixture(scope="session")
def fake_provider_fallback() -> Iterator[FakeProviderServer]:
    """Second ephemeral-port OpenAI-dialect fake for failover targets.

    DIFFERENT token counts than fake_provider (77/33 vs 120/45): usage
    assertions prove which server actually served a failover hop.
    """
    with FakeProviderServer(prompt_tokens=77, completion_tokens=33) as server:
        yield server


@pytest.fixture(scope="session")
def fake_provider_anthropic() -> Iterator[FakeAnthropicServer]:
    """Anthropic-dialect fake for the cross-dialect failover hop (88/44 tokens)."""
    with FakeAnthropicServer(prompt_tokens=88, completion_tokens=44) as server:
        yield server


@pytest.fixture(autouse=True)
def _reset_fake_provider(request: pytest.FixtureRequest) -> None:
    """Clear recorded requests/failures on the session servers between tests."""
    for name in (
        "fake_provider",
        "fake_provider_known_port",
        "fake_provider_fallback",
        "fake_provider_anthropic",
    ):
        if name in request.fixturenames:
            request.getfixturevalue(name).reset()


@pytest.fixture(scope="session")
def fake_provider_known_port() -> Iterator[FakeProviderServer]:
    """Fake provider on a conventional local port (1234/11434/8000) or skip."""
    server = start_on_conventional_port()
    if server is None:
        pytest.skip("conventional local ports 1234/11434/8000 all in use")
    yield server
    server.stop()


def expected_compat_provider(server: FakeProviderServer) -> str:
    """The compat provider name Solwyn should detect for *server*'s port."""
    return PORT_PROFILES.get(server.port, "openai_compatible")


class WireRecorder:
    """Records wire-bound payloads while delegating to the real implementations.

    Wraps two reporter seams on a constructed wrapper client (sync or async):
    - reporter.report_settlement -> .settlements ((confirm_request, event));
      the SINGLE settlement path for EVERY reservation-backed success —
      non-streaming chat/media AND streaming (PJ-1: settlement rides the
      reporter queue off the hot path; the enforcers have no confirm_cost).
      The confirm and its metadata event travel as one ordered unit.
    - reporter.report          -> .events (MetadataEvent, queued for ingest);
      the NON-settlement reports: a budget denial, a reservation-less success
      (e.g. an unpriced model), and every streaming/non-streaming ERROR (see
      client.py on_complete/on_error).

    Delegation is preserved, so everything still reaches the live API.
    """

    def __init__(self) -> None:
        self.budget_checks: list[dict[str, Any]] = []
        self.events: list[MetadataEvent] = []
        self.settlements: list[tuple[Any, MetadataEvent]] = []

    def attach(self, client: Any) -> WireRecorder:
        budget = client._solwyn_budget
        reporter = client._solwyn_reporter
        real_check = budget.check_budget
        real_report = reporter.report
        real_settlement = reporter.report_settlement

        if inspect.iscoroutinefunction(real_check):

            async def recording_check(**kwargs: Any) -> Any:
                self.budget_checks.append(dict(kwargs))
                return await real_check(**kwargs)

        else:

            def recording_check(**kwargs: Any) -> Any:
                self.budget_checks.append(dict(kwargs))
                return real_check(**kwargs)

        def recording_report(event: MetadataEvent) -> None:
            self.events.append(event)
            real_report(event)

        def recording_settlement(confirm_request: Any, event: MetadataEvent) -> None:
            self.settlements.append((confirm_request, event))
            real_settlement(confirm_request, event)

        budget.check_budget = recording_check
        reporter.report = recording_report
        reporter.report_settlement = recording_settlement
        return self


@pytest.fixture
def make_wrapped_client(
    test_credentials: Credentials, fake_provider: FakeProviderServer
) -> Iterator[Callable[..., Any]]:
    """Factory for real sync Solwyn(OpenAI(...)) wrappers against the harness.

    Defaults: fake_provider's ephemeral base_url (generic catch-all detection)
    and the session test project. Pass credentials=/base_url=/config kwargs to
    override. All created clients are closed on teardown (close() is safe to
    call twice, so tests may also close explicitly to assert flush behavior).
    """
    openai = pytest.importorskip("openai")
    created: list[Any] = []

    def _make(
        credentials: Credentials | None = None, base_url: str | None = None, **config: Any
    ) -> Any:
        creds = credentials or test_credentials
        inner = openai.OpenAI(
            base_url=base_url or fake_provider.base_url, api_key="sk-fake-provider-key"
        )
        client = Solwyn(
            inner,
            api_key=creds.api_key,
            api_url=creds.api_url,
            budget_check_cache_ttl=0,
            **config,
        )
        created.append(client)
        return client

    yield _make
    for client in created:
        client.close()


@pytest.fixture
async def make_async_wrapped_client(
    test_credentials: Credentials, fake_provider: FakeProviderServer
) -> AsyncIterator[Callable[..., Any]]:
    """Async factory: returns ENTERED AsyncSolwyn wrappers (reporter started).

    __aenter__ starts the async reporter's flush task; teardown closes every
    created client (idempotent with an explicit close inside the test).
    """
    openai = pytest.importorskip("openai")
    created: list[Any] = []

    async def _make(
        credentials: Credentials | None = None, base_url: str | None = None, **config: Any
    ) -> Any:
        creds = credentials or test_credentials
        inner = openai.AsyncOpenAI(
            base_url=base_url or fake_provider.base_url, api_key="sk-fake-provider-key"
        )
        client = AsyncSolwyn(
            inner,
            api_key=creds.api_key,
            api_url=creds.api_url,
            budget_check_cache_ttl=0,
            **config,
        )
        await client.__aenter__()
        created.append(client)
        return client

    yield _make
    for client in created:
        await client.close()
