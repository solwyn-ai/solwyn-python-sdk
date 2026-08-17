"""Live-API contract verification for the fields SDK behavior keys on.

The unit contract snapshot (tests/unit/test_contract_snapshot.py) pins only the
SDK side of the wire — server-side drift is invisible to it. These tests hit
the live API with the SDK's EXACT serialized check request (built through
``BudgetCheckRequest``, directive v1 opt-in included, mirroring
``_build_check_request``) and assert the response carries the two
behavior-bearing fields in the shapes the SDK reads:

* ``failover_directive.failover_tuning_allowed`` — entitlement gate for
  failover tuning; the SDK opts in via ``failover_directive_version="1"``.
* ``denied_by_period`` — OMITTED on allow (directive-v1 responses serialize
  exclude_none), present with the denying period on deny; the ``"agent_run"``
  and ``"run_stopped"`` literals key run-scoped handling, and the ``"tag"``
  literal keeps tag denials non-sticky (budget.py ``_cache_response``). Stopped
  runs must use ``mode="hard_deny"``, even for alert-only projects.

The PJ-2 lease block below does the same job for the lease wire: every
grant/renew field the SDK's admission ladder reads, every refusal status it
dispatches on (S2 classifies by STATUS alone — these pins are what make that
safe), the three endpoint paths budget.py hardcodes, and the lease-tagged
confirm.

If one of these fails against a running API, that is a cross-repo contract
bug — fix the drift, don't loosen the assertion.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from conftest import (
    Credentials,
    ProjectCredentials,
    _signup_token,
    budget_status,
    provision_project,
)

from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    BudgetMode,
    LeaseGrantRequest,
    LeaseGrantResponse,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    MetadataEvent,
    ProviderName,
)
from solwyn.budget import _LEASE_PATH, _LEASE_RENEW_PATH, _LEASE_SURRENDER_PATH

# Sized to gpt-5.5 prices ($5/M input, $30/M output): the first check's
# estimate (1000 input tokens + projected output) stays under the cap, and the
# burn below ($0.005 input + $0.06 output = $0.065) crosses it.
RUN_CAP_USD = 0.05
TAG_CAP_USD = 0.05


def _check_payload(
    agent_run_id: str | None = None,
    *,
    tags: dict[str, str] | None = None,
) -> dict[str, object]:
    """The SDK's exact check wire bytes: directive v1 opt-in, None-skipping."""
    request = BudgetCheckRequest(
        estimated_input_tokens=1000,
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        agent_run_id=agent_run_id,
        tags=tags,
        failover_directive_version="1",
    )
    return request.model_dump(mode="json")


def _post_check(credentials: Credentials, payload: dict[str, object]) -> dict[str, object]:
    with httpx.Client(base_url=credentials.api_url, timeout=10) as http:
        r = http.post(
            "/api/v1/budgets/check",
            json=payload,
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, dict):
        pytest.fail(f"budget check returned non-object JSON: {data!r}")
    return data


def _confirm_reservation(credentials: Credentials, reservation_id: str) -> None:
    """Settle enough gpt-5.5 usage to cross either $0.05 scoped cap."""
    with httpx.Client(base_url=credentials.api_url, timeout=10) as http:
        r = http.post(
            "/api/v1/budgets/confirm",
            json={
                "reservation_id": reservation_id,
                "model": "gpt-5.5",
                "provider": "openai",
                "call_id": str(uuid.uuid4()),
                "token_details": {"input_tokens": 1000, "output_tokens": 2000},
            },
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        r.raise_for_status()


@pytest.mark.integration
def test_metadata_ingest_accepts_parent_agent_run_id(
    test_credentials: Credentials,
) -> None:
    event = MetadataEvent(
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        input_tokens=10,
        output_tokens=5,
        latency_ms=12.0,
        status="success",
        is_model_fallback=False,
        sdk_instance_id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        agent_run_id=f"run-{uuid.uuid4().hex[:12]}",
        parent_agent_run_id=f"run-{uuid.uuid4().hex[:12]}",
        agent_run_name="child-contract-run",
        call_id=str(uuid.uuid4()),
    )

    with httpx.Client(base_url=test_credentials.api_url, timeout=10) as http:
        response = http.post(
            "/api/v1/metadata/ingest",
            json=[event.model_dump(mode="json")],
            headers={"Authorization": f"Bearer {test_credentials.api_key}"},
        )

    response.raise_for_status()
    assert response.status_code == 202
    assert response.json() == {"ingested": 1, "rejected": []}


@pytest.mark.integration
def test_local_metadata_ingest_accepts_complete_denial_receipt(api_url: str) -> None:
    """Pin C-1 without ever writing receipt fixtures to a remote control plane."""
    host = urlparse(api_url).hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        pytest.skip("denial-receipt live contract requires an explicitly local Core C-1 API")
    credentials = provision_project(
        api_url,
        name="sdk-denial-receipt-contract",
        budget_limit=100.0,
        budget_mode="alert_only",
    )
    event = MetadataEvent(
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        input_tokens=10,
        output_tokens=0,
        latency_ms=0.0,
        status="budget_denied",
        is_model_fallback=False,
        sdk_instance_id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        agent_run_id=f"run-{uuid.uuid4().hex[:12]}",
        call_id=str(uuid.uuid4()),
        deny_source="sticky_replay",
        deny_reason="monthly",
        denied_by_period="monthly",
        estimated_output_bound=512,
        velocity_flags=["repeat_size", "monotonic_growth"],
        receipt_aggregate_count=3,
    )

    with httpx.Client(base_url=credentials.api_url, timeout=10) as http:
        response = http.post(
            "/api/v1/metadata/ingest",
            json=[event.model_dump(mode="json")],
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )

    response.raise_for_status()
    assert response.status_code == 202
    assert response.json() == {"ingested": 1, "rejected": []}


@pytest.fixture(scope="session")
def run_capped_credentials(api_url: str) -> Credentials:
    """A hard_deny project with a $0.02 runaway-run cap and ample project budget.

    Bootstraps its own account (the runaway-run rule is a budget-threshold PUT,
    JWT-only) so run-cap denials cannot collide with the shared test project.
    """
    session_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        jwt_headers = {"Authorization": f"Bearer {token}"}
        r = http.post(
            "/api/v1/projects",
            json={
                "name": f"sdk-runcap-{session_id}",
                "budget_limit": 100.0,
                "budget_period": "monthly",
                "budget_mode": "hard_deny",
            },
            headers=jwt_headers,
        )
        r.raise_for_status()
        project_id = r.json()["id"]
        key = r.json()["key"]

        r = http.put(
            f"/api/v1/projects/{project_id}/budget",
            json={
                "limit_usd": 100.0,
                "period": "monthly",
                "mode": "hard_deny",
                "thresholds": [
                    {
                        "type": "runaway_run",
                        "spend_threshold_usd": RUN_CAP_USD,
                        "channel_ids": [],
                    }
                ],
            },
            headers=jwt_headers,
        )
        if r.status_code != 200:
            pytest.fail(f"runaway_run budget rule PUT failed: {r.status_code} {r.text}")
    return Credentials(api_url=api_url, api_key=key)


@pytest.fixture(scope="session")
def tag_capped_credentials(api_url: str) -> Credentials:
    """A hard_deny project with a $0.05 customer=acme tag cap."""
    session_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        jwt_headers = {"Authorization": f"Bearer {token}"}
        r = http.post(
            "/api/v1/projects",
            json={
                "name": f"sdk-tagcap-{session_id}",
                "budget_limit": 100.0,
                "budget_period": "monthly",
                "budget_mode": "hard_deny",
            },
            headers=jwt_headers,
        )
        r.raise_for_status()
        project_id = r.json()["id"]
        key = r.json()["key"]

        r = http.put(
            f"/api/v1/projects/{project_id}/budget",
            json={
                "limit_usd": 100.0,
                "period": "monthly",
                "mode": "hard_deny",
                "thresholds": [
                    {
                        "type": "scoped_budget",
                        "scope": "tag",
                        "match": "customer",
                        "match_value": "acme",
                        "limit": TAG_CAP_USD,
                        "mode": "hard_deny",
                        "channel_ids": [],
                    }
                ],
            },
            headers=jwt_headers,
        )
        if r.status_code != 200:
            pytest.fail(f"tag-scoped budget rule PUT failed: {r.status_code} {r.text}")
    return Credentials(api_url=api_url, api_key=key)


@pytest.fixture
def stopped_run_credentials(api_url: str) -> tuple[Credentials, str]:
    """An alert-only project with one dashboard-stopped explicit run."""
    session_id = uuid.uuid4().hex[:12]
    raw_run_id = f"run-{uuid.uuid4().hex[:12]}"
    run_name = f"stopped-contract-{session_id}"
    with httpx.Client(base_url=api_url, timeout=10) as http:
        token = _signup_token(http, session_id)
        jwt_headers = {"Authorization": f"Bearer {token}"}
        project_response = http.post(
            "/api/v1/projects",
            json={
                "name": f"sdk-stopped-run-{session_id}",
                "budget_limit": 100.0,
                "budget_period": "monthly",
                "budget_mode": "alert_only",
            },
            headers=jwt_headers,
        )
        project_response.raise_for_status()
        project = project_response.json()
        project_id = project["id"]
        credentials = Credentials(api_url=api_url, api_key=project["key"])
        event = MetadataEvent(
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            input_tokens=10,
            output_tokens=5,
            latency_ms=12.0,
            status="success",
            is_model_fallback=False,
            sdk_instance_id=uuid.uuid4().hex,
            timestamp=datetime.now(UTC),
            agent_run_id=raw_run_id,
            agent_run_name=run_name,
            call_id=str(uuid.uuid4()),
        )
        ingest = http.post(
            "/api/v1/metadata/ingest",
            json=[event.model_dump(mode="json")],
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        )
        ingest.raise_for_status()
        runs_response = http.get(
            f"/api/v1/projects/{project_id}/agent-runs",
            params={"q": run_name},
            headers=jwt_headers,
        )
        runs_response.raise_for_status()
        runs = runs_response.json()["runs"]
        matching_runs = [run for run in runs if run["name"] == run_name]
        assert len(matching_runs) == 1, runs
        stop = http.post(
            f"/api/v1/projects/{project_id}/agent-runs/{matching_runs[0]['id']}/stop",
            headers=jwt_headers,
        )
        stop.raise_for_status()

    return credentials, raw_run_id


@pytest.mark.integration
class TestLiveFailoverDirectiveContract:
    @pytest.mark.integration
    def test_allow_response_carries_v1_directive_with_tuning_entitlement(
        self, test_credentials: Credentials
    ) -> None:
        payload = _post_check(test_credentials, _check_payload())

        assert payload["allowed"] is True
        directive = payload.get("failover_directive")
        assert isinstance(directive, dict), (
            f"live API omitted failover_directive on a directive-v1 check: {payload}"
        )
        assert directive["version"] == "1"
        # The entitlement gate the SDK reads (#33): a real boolean, not a
        # truthy stand-in — json.loads only produces bool for JSON booleans.
        assert isinstance(directive["failover_tuning_allowed"], bool)

        parsed = BudgetCheckResponse.model_validate(payload)
        assert parsed.failover_directive is not None
        assert (
            parsed.failover_directive.failover_tuning_allowed
            == directive["failover_tuning_allowed"]
        )


@pytest.mark.integration
class TestLiveDeniedByPeriodContract:
    @pytest.mark.integration
    def test_allow_response_omits_denied_by_period(self, test_credentials: Credentials) -> None:
        # Directive-v1 responses serialize exclude_none: an allow response has
        # NO denied_by_period key. This is the live behavior that makes the
        # SDK-side default (Field(None)) the correct posture — see
        # test_contract_snapshot.test_directive_v1_allow_response_parses_without_denied_by_period.
        payload = _post_check(test_credentials, _check_payload())

        assert payload["allowed"] is True
        assert "denied_by_period" not in payload
        assert BudgetCheckResponse.model_validate(payload).denied_by_period is None

    @pytest.mark.integration
    def test_project_period_denial_carries_denied_by_period(
        self, hard_denied_credentials: Credentials
    ) -> None:
        payload = _post_check(hard_denied_credentials, _check_payload())

        assert payload["allowed"] is False
        assert payload.get("denied_by_period") == "monthly", (
            f"live API deny did not name the denying period: {payload}"
        )
        parsed = BudgetCheckResponse.model_validate(payload)
        assert parsed.denied_by_period == "monthly"

    @pytest.mark.integration
    def test_run_cap_denial_reports_agent_run_and_stays_run_scoped(
        self, run_capped_credentials: Credentials
    ) -> None:
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        first = _post_check(run_capped_credentials, _check_payload(agent_run_id=run_id))
        assert first["allowed"] is True
        reservation_id = first.get("reservation_id")
        assert isinstance(reservation_id, str), f"expected a reservation: {first}"

        # Burn past the $0.05 run cap: $0.005 input + $0.06 output = $0.065.
        _confirm_reservation(run_capped_credentials, reservation_id)

        denied = _post_check(run_capped_credentials, _check_payload(agent_run_id=run_id))
        assert denied["allowed"] is False
        assert denied.get("denied_by_period") == "agent_run", (
            "run-cap denial must carry the exact literal run-scoped sticky "
            f"denial keys on (budget.py); live API returned: {denied}"
        )
        assert BudgetCheckResponse.model_validate(denied).denied_by_period == "agent_run"

        # The denial is scoped to THIS run: a fresh run id on the same project
        # is still allowed (the shape the SDK's per-run stickiness relies on).
        fresh = _post_check(
            run_capped_credentials, _check_payload(agent_run_id=f"run-{uuid.uuid4().hex[:12]}")
        )
        assert fresh["allowed"] is True

    @pytest.mark.integration
    def test_stopped_run_denial_reports_hard_deny_run_stopped(
        self, stopped_run_credentials: tuple[Credentials, str]
    ) -> None:
        credentials, raw_run_id = stopped_run_credentials

        denied = _post_check(credentials, _check_payload(agent_run_id=raw_run_id))

        assert denied["allowed"] is False
        assert denied.get("mode") == "hard_deny", (
            f"dashboard stops must override an alert-only project's mode: {denied}"
        )
        assert denied.get("denied_by_period") == "run_stopped", (
            "the SDK's typed stopped-run error keys on the exact run_stopped "
            f"literal; live API returned: {denied}"
        )
        parsed = BudgetCheckResponse.model_validate(denied)
        assert parsed.mode is BudgetMode.HARD_DENY
        assert parsed.denied_by_period == "run_stopped"

    @pytest.mark.integration
    def test_tag_cap_denial_reports_tag_and_remains_selector_scoped(
        self, tag_capped_credentials: Credentials
    ) -> None:
        tags = {"customer": "acme"}
        first = _post_check(tag_capped_credentials, _check_payload(tags=tags))
        assert first["allowed"] is True
        reservation_id = first.get("reservation_id")
        assert isinstance(reservation_id, str), f"expected a reservation: {first}"

        # Burn past the $0.05 tag cap: $0.005 input + $0.06 output = $0.065.
        _confirm_reservation(tag_capped_credentials, reservation_id)

        denied = _post_check(tag_capped_credentials, _check_payload(tags=tags))
        assert denied["allowed"] is False
        assert denied.get("denied_by_period") == "tag", (
            "tag-cap denial must carry the exact literal the SDK's non-sticky "
            f"branch keys on (budget.py); live API returned: {denied}"
        )
        assert BudgetCheckResponse.model_validate(denied).denied_by_period == "tag"

        # A tag denial is selector-scoped, not project-sticky: untagged traffic
        # on the same project remains allowed.
        untagged = _post_check(tag_capped_credentials, _check_payload())
        assert untagged["allowed"] is True


# ── PJ-2 budget leases (Task S5) ─────────────────────────────────────────────
#
# One assertion block per wire shape. Every request goes out as the SDK's own
# serialized bytes (``LeaseGrantRequest``/``LeaseRenewRequest``/
# ``LeaseSurrenderRequest``/``BudgetConfirmRequest``) and every response is
# checked BOTH as raw JSON (so an exclude-none omission is visible as a missing
# key) and through ``LeaseGrantResponse.model_validate`` (so SDK-side drift is
# visible too).

LEASE_MODEL = "gpt-5.5"
# Core issues ``lse_`` + token_urlsafe(24); both repos bound the field at 64.
LEASE_ID_MAX_LENGTH = 64
# The keys that exist ONLY on a grant that carries a lease.
LEASE_BLOCK_KEYS = (
    "lease_id",
    "generation",
    "granted_tokens",
    "refresh_interval_s",
    "lease_length_s",
    "headroom_share_tokens",
    "posture",
    "final_grant",
)


def _lease_headers(credentials: Credentials) -> dict[str, str]:
    return {"Authorization": f"Bearer {credentials.api_key}"}


def _post_lease(credentials: Credentials, path: str, payload: dict[str, object]) -> httpx.Response:
    with httpx.Client(base_url=credentials.api_url, timeout=15) as http:
        return http.post(path, json=payload, headers=_lease_headers(credentials))


def _grant_payload(
    run_id: str, holder_id: str, *, model: str = LEASE_MODEL, fail_open: bool = True
) -> dict[str, Any]:
    """The SDK's exact grant wire bytes (mirrors ``_build_grant_request``)."""
    return LeaseGrantRequest(
        agent_run_id=run_id,
        holder_id=holder_id,
        model=model,
        provider=ProviderName.OPENAI,
        fallback_providers=[],
        fallback_models=[],
        fail_open=fail_open,
        estimated_input_tokens=1000,
    ).model_dump(mode="json")


def _grant(credentials: Credentials, run_id: str, holder_id: str, **kwargs: Any) -> dict[str, Any]:
    response = _post_lease(credentials, _LEASE_PATH, _grant_payload(run_id, holder_id, **kwargs))
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        pytest.fail(f"lease grant returned non-object JSON: {body!r}")
    return body


def _renew(
    credentials: Credentials, lease_id: str, holder_id: str, generation: int
) -> httpx.Response:
    payload = LeaseRenewRequest(
        lease_id=lease_id, holder_id=holder_id, generation=generation
    ).model_dump(mode="json")
    return _post_lease(credentials, _LEASE_RENEW_PATH, payload)


def _surrender(
    credentials: Credentials, lease_id: str, holder_id: str, generation: int
) -> httpx.Response:
    payload = LeaseSurrenderRequest(
        lease_id=lease_id, holder_id=holder_id, generation=generation
    ).model_dump(mode="json")
    return _post_lease(credentials, _LEASE_SURRENDER_PATH, payload)


def _run_id() -> str:
    return f"lease-contract-{uuid.uuid4().hex[:12]}"


def _holder_id() -> str:
    return f"holder-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def lease_contract_project(api_url: str) -> ProjectCredentials:
    """A fresh alert_only project with ample budget, per test.

    Per test, not per session: a held lease claims budget while it is held, so a
    shared project would make one pin's leftovers another pin's headroom.
    """
    return provision_project(
        api_url, name="sdk-lease-contract", budget_limit=20.0, budget_mode="alert_only"
    )


@pytest.mark.integration
class TestLiveLeaseGrantContract:
    @pytest.mark.integration
    def test_grant_carries_every_behavior_bearing_field(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        holder = _holder_id()
        payload = _grant(lease_contract_project, _run_id(), holder)

        assert payload["eligible"] is True
        assert payload["allowed"] is True
        # Refusals only: an eligible allow never explains itself.
        assert "ineligible_reason" not in payload
        assert "denied_by_period" not in payload

        lease_id = payload["lease_id"]
        assert isinstance(lease_id, str)
        assert 0 < len(lease_id) <= LEASE_ID_MAX_LENGTH
        assert payload["generation"] == 1, "a fresh grant starts at generation 1"
        assert isinstance(payload["granted_tokens"], int)
        assert payload["granted_tokens"] > 0
        assert payload["refresh_interval_s"] > 0
        assert payload["lease_length_s"] > 0
        assert payload["refresh_interval_s"] < payload["lease_length_s"], (
            "a holder must get at least one renewal window inside its lease"
        )
        assert isinstance(payload["headroom_share_tokens"], int)
        assert payload["headroom_share_tokens"] > 0
        assert payload["posture"] == {"mode": "alert_only", "on_unreachable": "fail_open"}
        assert payload["final_grant"] is False

        # The always-present display snapshot the SDK logs a last-known
        # position from (never computed SDK-side).
        assert isinstance(payload["project_id"], str)
        assert payload["mode"] == "alert_only"
        assert payload["budget_limit"] == 20.0
        assert isinstance(payload["current_usage"], float)
        assert isinstance(payload["remaining_budget"], float)

        parsed = LeaseGrantResponse.model_validate(payload)
        assert parsed.lease_id == lease_id
        assert parsed.posture is not None
        assert parsed.posture.mode is BudgetMode.ALERT_ONLY
        assert parsed.posture.on_unreachable == "fail_open"
        assert parsed.final_grant is False

        _surrender(lease_contract_project, lease_id, holder, payload["generation"])

    @pytest.mark.integration
    def test_grant_echoes_the_clients_unreachable_posture(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        # posture.on_unreachable is the ladder's input for "share exhausted and
        # Solwyn unreachable": it must echo the client's own fail_open.
        holder = _holder_id()
        payload = _grant(lease_contract_project, _run_id(), holder, fail_open=False)

        assert payload["posture"]["on_unreachable"] == "local_enforce"
        _surrender(lease_contract_project, payload["lease_id"], holder, payload["generation"])

    @pytest.mark.integration
    def test_grant_on_a_hard_deny_project_echoes_hard_deny_posture(self, api_url: str) -> None:
        """posture.mode is the customer's cap verdict, and it must not drift.

        The §4 ladder keys share-exhaustion entirely off ``posture.mode``:
        silent drift to ``alert_only`` would keep spending a hard_deny
        customer's money during exactly the outage their cap exists for. Every
        other live grant pin runs alert_only — this is the negative space, on
        an UNDER-cap hard_deny project (so the grant carries a real lease block
        rather than the deny shape pinned below).
        """
        credentials = provision_project(
            api_url, name="sdk-lease-posture-deny", budget_limit=100.0, budget_mode="hard_deny"
        )
        holder = _holder_id()
        payload = _grant(credentials, _run_id(), holder)

        assert payload["eligible"] is True
        assert payload["allowed"] is True, "an under-cap hard_deny project still gets a lease"
        assert payload["posture"]["mode"] == "hard_deny", (
            f"the grant must echo the project's hard_deny posture: {payload}"
        )
        assert payload["mode"] == "hard_deny"

        parsed = LeaseGrantResponse.model_validate(payload)
        assert parsed.posture is not None
        assert parsed.posture.mode is BudgetMode.HARD_DENY

        _surrender(credentials, payload["lease_id"], holder, payload["generation"])

    @pytest.mark.integration
    def test_ineligible_grant_omits_the_lease_block(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        # A model tokens cannot price cannot denominate a grant. The SDK marks
        # the run ineligible and uses the legacy path — it must never find a
        # lease block (or a null one) on this shape.
        payload = _grant(
            lease_contract_project, _run_id(), _holder_id(), model="no-such-model-for-leases"
        )

        assert payload["eligible"] is False
        assert payload["ineligible_reason"] == "zero_rate_model"
        assert payload["allowed"] is True, "ineligible is not a denial"
        for key in LEASE_BLOCK_KEYS:
            assert key not in payload, f"ineligible response leaked lease key {key}: {payload}"
        assert LeaseGrantResponse.model_validate(payload).lease_id is None

    @pytest.mark.integration
    def test_deny_grant_omits_the_lease_block_and_reason(
        self, hard_denied_credentials: Credentials
    ) -> None:
        payload = _grant(hard_denied_credentials, _run_id(), _holder_id())

        assert payload["eligible"] is True
        assert payload["allowed"] is False
        assert payload["denied_by_period"] == "monthly", (
            f"a lease deny must name the denying period like a check does: {payload}"
        )
        assert "ineligible_reason" not in payload
        for key in LEASE_BLOCK_KEYS:
            assert key not in payload, f"deny response leaked lease key {key}: {payload}"

        parsed = LeaseGrantResponse.model_validate(payload)
        assert parsed.allowed is False
        assert parsed.denied_by_period == "monthly"
        assert parsed.lease_id is None


@pytest.mark.integration
class TestLiveLeaseRefusalContract:
    """The statuses S2 dispatches on (status-only classification, no body read)."""

    @pytest.mark.integration
    def test_holder_cap_is_409_lease_holder_cap_exceeded(self, api_url: str) -> None:
        credentials = provision_project(
            api_url, name="sdk-lease-cap", budget_limit=20.0, budget_mode="alert_only"
        )
        run_id = _run_id()
        refusal = None
        for index in range(40):
            response = _post_lease(
                credentials, _LEASE_PATH, _grant_payload(run_id, f"cap-holder-{index}")
            )
            if response.status_code != 200:
                refusal = response
                break

        assert refusal is not None, "the active-holder cap never fired within 40 holders"
        assert refusal.status_code == 409
        assert refusal.json()["detail"]["code"] == "lease_holder_cap_exceeded"

    @pytest.mark.integration
    def test_unknown_lease_renew_is_404_lease_not_found(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        response = _renew(lease_contract_project, "lse_not-a-real-lease", _holder_id(), 1)

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "lease_not_found"

    @pytest.mark.integration
    def test_unfenceable_generation_is_409_lease_generation_conflict(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        holder = _holder_id()
        grant = _grant(lease_contract_project, _run_id(), holder)

        response = _renew(lease_contract_project, grant["lease_id"], holder, 99)

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "lease_generation_conflict"
        _surrender(lease_contract_project, grant["lease_id"], holder, grant["generation"])


@pytest.mark.integration
class TestLiveLeaseRenewalContract:
    @pytest.mark.integration
    def test_renew_echoing_g_returns_g_plus_one(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        holder = _holder_id()
        grant = _grant(lease_contract_project, _run_id(), holder)

        response = _renew(lease_contract_project, grant["lease_id"], holder, grant["generation"])
        response.raise_for_status()
        renewed = response.json()

        assert renewed["generation"] == grant["generation"] + 1
        assert renewed["lease_id"] == grant["lease_id"]
        assert renewed["eligible"] is True
        assert renewed["allowed"] is True
        assert isinstance(renewed["granted_tokens"], int)
        assert renewed["posture"]["mode"] == "alert_only"
        LeaseGrantResponse.model_validate(renewed)

        _surrender(lease_contract_project, grant["lease_id"], holder, renewed["generation"])

    @pytest.mark.integration
    def test_surrender_returns_released_tokens_only(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        holder = _holder_id()
        grant = _grant(lease_contract_project, _run_id(), holder)

        response = _surrender(
            lease_contract_project, grant["lease_id"], holder, grant["generation"]
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"released_tokens"}
        assert isinstance(body["released_tokens"], int)
        assert body["released_tokens"] > 0

        # Idempotent: a retried release from an exiting process is not an error.
        again = _surrender(lease_contract_project, grant["lease_id"], holder, grant["generation"])
        assert again.status_code == 200
        assert again.json() == {"released_tokens": 0}


@pytest.mark.integration
class TestLiveLeaseEndpointPaths:
    @pytest.mark.integration
    def test_sdk_hardcoded_paths_are_the_live_routes(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        # budget.py hardcodes these three strings; a router rename would show up
        # here as a 404 with FastAPI's string detail rather than a lease code.
        assert _LEASE_PATH == "/api/v1/budgets/lease"
        assert _LEASE_RENEW_PATH == "/api/v1/budgets/lease/renew"
        assert _LEASE_SURRENDER_PATH == "/api/v1/budgets/lease/surrender"

        holder = _holder_id()
        grant = _post_lease(lease_contract_project, _LEASE_PATH, _grant_payload(_run_id(), holder))
        assert grant.status_code == 200
        body = grant.json()

        renew = _renew(lease_contract_project, body["lease_id"], holder, body["generation"])
        assert renew.status_code == 200

        surrender = _surrender(
            lease_contract_project, body["lease_id"], holder, renew.json()["generation"]
        )
        assert surrender.status_code == 200


@pytest.mark.integration
class TestLiveLeaseConfirmContract:
    @pytest.mark.integration
    def test_lease_tagged_confirm_settles_and_replays_without_double_settling(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        holder = _holder_id()
        grant = _grant(lease_contract_project, _run_id(), holder)
        lease_id = grant["lease_id"]

        confirm = BudgetConfirmRequest(
            lease_id=lease_id,
            model=LEASE_MODEL,
            provider=ProviderName.OPENAI,
            call_id=str(uuid.uuid4()),
            # Deliberately past what the grant's claim covers: spend INSIDE the
            # claim is already counted, so only an overshoot is observable in
            # the project's usage.
            token_details=TokenDetails(input_tokens=100_000, output_tokens=500_000),
        )
        wire = confirm.model_dump(mode="json")
        assert "reservation_id" not in wire, (
            "the settlement key is exclusive on the wire: a lease-settled confirm "
            f"must carry no reservation_id: {wire}"
        )
        assert wire["lease_id"] == lease_id

        before = budget_status(lease_contract_project)["current_usage"]
        response = _post_lease(lease_contract_project, "/api/v1/budgets/confirm", wire)
        assert response.status_code == 204, response.text
        settled = budget_status(lease_contract_project)["current_usage"]
        assert settled > before, (
            f"a lease-tagged confirm past the claim never reached the budget: {before} -> {settled}"
        )

        # C3 replay: the same call_id settles nothing a second time.
        replay = _post_lease(lease_contract_project, "/api/v1/budgets/confirm", wire)
        assert replay.status_code == 204, replay.text
        assert budget_status(lease_contract_project)["current_usage"] == settled, (
            "a replayed lease-tagged confirm settled the same call twice"
        )

        _surrender(lease_contract_project, lease_id, holder, grant["generation"])

    @pytest.mark.integration
    def test_confirm_requires_exactly_one_settlement_key(
        self, lease_contract_project: ProjectCredentials
    ) -> None:
        # The SDK model refuses to BUILD either shape (model validator), so the
        # pin posts raw bytes: what is being verified is that the server refuses
        # them too, which is what makes the exclusive key a wire contract rather
        # than an SDK convention.
        base: dict[str, Any] = {
            "model": LEASE_MODEL,
            "provider": "openai",
            "call_id": str(uuid.uuid4()),
            "token_details": {"input_tokens": 10, "output_tokens": 10},
        }
        holder = _holder_id()
        grant = _grant(lease_contract_project, _run_id(), holder)

        both = _post_lease(
            lease_contract_project,
            "/api/v1/budgets/confirm",
            {**base, "lease_id": grant["lease_id"], "reservation_id": "res_not_real"},
        )
        assert both.status_code == 422, both.text

        neither = _post_lease(lease_contract_project, "/api/v1/budgets/confirm", dict(base))
        assert neither.status_code == 422, neither.text

        _surrender(lease_contract_project, grant["lease_id"], holder, grant["generation"])
