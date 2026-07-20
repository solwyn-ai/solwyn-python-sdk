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
  literal keys run-scoped sticky denial (budget.py ``_cache_response``).

If one of these fails against a running API, that is a cross-repo contract
bug — fix the drift, don't loosen the assertion.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from conftest import Credentials, _signup_token

from solwyn._types import BudgetCheckRequest, BudgetCheckResponse, ProviderName

# Sized to gpt-5.5 prices ($5/M input, $30/M output): the first check's
# estimate (1000 input tokens + projected output) stays under the cap, and the
# burn below ($0.005 input + $0.06 output = $0.065) crosses it.
RUN_CAP_USD = 0.05


def _check_payload(agent_run_id: str | None = None) -> dict[str, object]:
    """The SDK's exact check wire bytes: directive v1 opt-in, None-skipping."""
    request = BudgetCheckRequest(
        estimated_input_tokens=1000,
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        agent_run_id=agent_run_id,
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
        with httpx.Client(base_url=run_capped_credentials.api_url, timeout=10) as http:
            r = http.post(
                "/api/v1/budgets/confirm",
                json={
                    "reservation_id": reservation_id,
                    "model": "gpt-5.5",
                    "provider": "openai",
                    "call_id": f"live-contract-burn-{run_id}",
                    "token_details": {"input_tokens": 1000, "output_tokens": 2000},
                },
                headers={"Authorization": f"Bearer {run_capped_credentials.api_key}"},
            )
            r.raise_for_status()

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
