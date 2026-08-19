"""Run the shared behavior-bearing wire contract against the live API."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from conftest import (
    Credentials,
    ProjectCredentials,
    _signup_token,
    provision_project,
)

from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetConfirmRequest,
    LeaseGrantRequest,
    MetadataEvent,
    ProviderName,
)
from solwyn.testing.contract import (
    assert_check_contract,
    assert_confirm_contract,
    assert_lease_contract,
    assert_receipt_ingest_contract,
    assert_run_control_contract,
)

_SCOPED_CAP_USD = 0.05


@dataclass(frozen=True)
class _CheckContractCredentials:
    allow: Credentials
    monthly: Credentials
    agent_run: Credentials
    stopped: Credentials
    tag: Credentials


def _create_scoped_project(
    api_url: str,
    *,
    name: str,
    threshold: dict[str, object],
) -> Credentials:
    session_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        headers = {"Authorization": f"Bearer {token}"}
        created = http.post(
            "/api/v1/projects",
            json={
                "name": f"{name}-{session_id}",
                "budget_limit": 100.0,
                "budget_period": "monthly",
                "budget_mode": "hard_deny",
            },
            headers=headers,
        )
        created.raise_for_status()
        project = created.json()
        updated = http.put(
            f"/api/v1/projects/{project['id']}/budget",
            json={
                "limit_usd": 100.0,
                "period": "monthly",
                "mode": "hard_deny",
                "thresholds": [threshold],
            },
            headers=headers,
        )
        updated.raise_for_status()
    return Credentials(api_url=api_url, api_key=project["key"])


def _burn_scoped_budget(
    credentials: Credentials,
    *,
    agent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    check = BudgetCheckRequest(
        estimated_input_tokens=1000,
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        agent_run_id=agent_run_id,
        tags=tags,
        failover_directive_version="1",
    )
    headers = {"Authorization": f"Bearer {credentials.api_key}"}
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        checked = http.post(
            "/api/v1/budgets/check",
            json=check.model_dump(mode="json"),
            headers=headers,
        )
        checked.raise_for_status()
        reservation_id = checked.json().get("reservation_id")
        if not isinstance(reservation_id, str):
            pytest.fail(f"shared contract burn did not receive a reservation: {checked.json()}")
        confirmed = http.post(
            "/api/v1/budgets/confirm",
            json=BudgetConfirmRequest(
                reservation_id=reservation_id,
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                call_id=str(uuid.uuid4()),
                token_details=TokenDetails(input_tokens=1000, output_tokens=2000),
            ).model_dump(mode="json"),
            headers=headers,
        )
        confirmed.raise_for_status()


def _create_stopped_project(api_url: str, *, run_id: str = "contract-stopped-run") -> Credentials:
    session_id = uuid.uuid4().hex[:12]
    run_name = f"contract-stopped-{session_id}"
    with httpx.Client(base_url=api_url, timeout=15) as http:
        token = _signup_token(http, session_id)
        headers = {"Authorization": f"Bearer {token}"}
        created = http.post(
            "/api/v1/projects",
            json={
                "name": f"contract-stopped-{session_id}",
                "budget_limit": 100.0,
                "budget_period": "monthly",
                "budget_mode": "alert_only",
            },
            headers=headers,
        )
        created.raise_for_status()
        project = created.json()
        credentials = Credentials(api_url=api_url, api_key=project["key"])
        ingested = http.post(
            "/api/v1/metadata/ingest",
            json=[
                MetadataEvent(
                    model="gpt-5.5",
                    provider=ProviderName.OPENAI,
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=1.0,
                    status="success",
                    is_model_fallback=False,
                    sdk_instance_id=uuid.uuid4().hex,
                    timestamp=datetime.now(UTC),
                    agent_run_id=run_id,
                    agent_run_name=run_name,
                    call_id=str(uuid.uuid4()),
                ).model_dump(mode="json")
            ],
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        ingested.raise_for_status()
        listed = http.get(
            f"/api/v1/projects/{project['id']}/agent-runs",
            params={"q": run_name},
            headers=headers,
        )
        listed.raise_for_status()
        matches = [run for run in listed.json()["runs"] if run["name"] == run_name]
        if len(matches) != 1:
            pytest.fail(f"shared contract stopped-run setup was ambiguous: {matches!r}")
        stopped = http.post(
            f"/api/v1/projects/{project['id']}/agent-runs/{matches[0]['id']}/stop",
            headers=headers,
        )
        stopped.raise_for_status()
    return credentials


@pytest.fixture
def check_contract_credentials(
    api_url: str,
    test_credentials: Credentials,
    hard_denied_credentials: Credentials,
) -> _CheckContractCredentials:
    run_credentials = _create_scoped_project(
        api_url,
        name="sdk-shared-run-contract",
        threshold={
            "type": "runaway_run",
            "spend_threshold_usd": _SCOPED_CAP_USD,
            # Core requires an explicit mode on every new/changed runaway_run
            # rule; the run-scoped denial this fixture provokes is a hard deny.
            "mode": "hard_deny",
            "channel_ids": [],
        },
    )
    _burn_scoped_budget(run_credentials, agent_run_id="contract-agent-run")

    tag_credentials = _create_scoped_project(
        api_url,
        name="sdk-shared-tag-contract",
        threshold={
            "type": "scoped_budget",
            "scope": "tag",
            "match": "customer",
            "match_value": "acme",
            "limit": _SCOPED_CAP_USD,
            "mode": "hard_deny",
            "channel_ids": [],
        },
    )
    _burn_scoped_budget(tag_credentials, tags={"customer": "acme"})

    return _CheckContractCredentials(
        allow=test_credentials,
        monthly=hard_denied_credentials,
        agent_run=run_credentials,
        stopped=_create_stopped_project(api_url),
        tag=tag_credentials,
    )


def _request_body(request: httpx.Request) -> dict[str, Any]:
    if not request.content:
        return {}
    body = json.loads(request.content)
    if not isinstance(body, dict):
        return {}
    return body


def _route_check_contract(
    credentials: _CheckContractCredentials,
) -> Callable[[httpx.Request], None]:
    def route(request: httpx.Request) -> None:
        if request.url.path != "/api/v1/budgets/check":
            return
        body = _request_body(request)
        tags = body.get("tags")
        run_id = body.get("agent_run_id")
        selected = credentials.allow
        if isinstance(tags, dict) and tags.get("contract_case") == "monthly":
            selected = credentials.monthly
        elif run_id == "contract-stopped-run":
            selected = credentials.stopped
        elif isinstance(tags, dict) and tags.get("customer") == "acme":
            selected = credentials.tag
        elif run_id == "contract-agent-run":
            selected = credentials.agent_run
        request.headers["Authorization"] = f"Bearer {selected.api_key}"

    return route


@pytest.mark.integration
def test_shared_check_contract_against_live(
    check_contract_credentials: _CheckContractCredentials,
) -> None:
    credentials = check_contract_credentials
    with httpx.Client(
        base_url=credentials.allow.api_url,
        timeout=15,
        event_hooks={"request": [_route_check_contract(credentials)]},
    ) as http:
        assert_check_contract(http, credentials.allow.api_key)


@pytest.mark.integration
def test_shared_run_control_contract_against_live(api_url: str) -> None:
    # Same provisioning shape as test_live_contract.StoppableRun: ingest one
    # event to mint the run, resolve its stored id, then stop it over the
    # dashboard JWT before the pack reads the wire.
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    credentials = _create_stopped_project(api_url, run_id=run_id)

    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        assert_run_control_contract(http, credentials.api_key, stopped_run_id=run_id)


@pytest.mark.integration
def test_shared_receipt_ingest_contract_against_live(api_url: str) -> None:
    host = urlparse(api_url).hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        pytest.skip("denial-receipt live contract requires an explicitly local Core C-1 API")
    credentials = provision_project(
        api_url,
        name="sdk-shared-receipt-contract",
        budget_limit=100.0,
        budget_mode="alert_only",
    )

    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        assert_receipt_ingest_contract(http, credentials.api_key)


@pytest.mark.integration
def test_shared_confirm_contract_against_live(test_credentials: Credentials) -> None:
    with httpx.Client(base_url=test_credentials.api_url, timeout=15) as http:
        assert_confirm_contract(http, test_credentials.api_key)


def _fill_holder_cap(credentials: ProjectCredentials) -> None:
    headers = {"Authorization": f"Bearer {credentials.api_key}"}
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        refusal: httpx.Response | None = None
        for index in range(40):
            response = http.post(
                "/api/v1/budgets/lease",
                json=LeaseGrantRequest(
                    agent_run_id="contract-holder-cap",
                    holder_id=f"contract-prefill-{index}",
                    model="gpt-5.5",
                    provider=ProviderName.OPENAI,
                    fail_open=True,
                    estimated_input_tokens=1000,
                ).model_dump(mode="json"),
                headers=headers,
            )
            if response.status_code != 200:
                refusal = response
                break
        if refusal is None:
            pytest.fail("shared contract holder cap never fired within 40 holders")
        if refusal.status_code != 409:
            pytest.fail(f"shared contract holder cap setup returned {refusal.status_code}")


@pytest.mark.integration
def test_shared_lease_contract_against_live(
    api_url: str,
    hard_denied_credentials: Credentials,
) -> None:
    eligible = provision_project(
        api_url,
        name="sdk-shared-lease-contract",
        budget_limit=20.0,
        budget_mode="alert_only",
    )
    holder_cap = provision_project(
        api_url,
        name="sdk-shared-holder-cap",
        budget_limit=20.0,
        budget_mode="alert_only",
    )
    _fill_holder_cap(holder_cap)

    def route(request: httpx.Request) -> None:
        if request.url.path != "/api/v1/budgets/lease":
            return
        run_id = _request_body(request).get("agent_run_id")
        if run_id == "contract-holder-cap":
            selected = holder_cap
        elif run_id == "contract-lease-denied":
            selected = hard_denied_credentials
        else:
            selected = eligible
        request.headers["Authorization"] = f"Bearer {selected.api_key}"

    with httpx.Client(
        base_url=api_url,
        timeout=15,
        event_hooks={"request": [route]},
    ) as http:
        assert_lease_contract(http, eligible.api_key)
