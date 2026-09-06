"""Shared control-plane contract pack against the in-process double."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from solwyn._types import LeaseGrantRequest, ProviderName
from solwyn.testing import FakeControlPlane
from solwyn.testing.contract import (
    assert_check_contract,
    assert_confirm_contract,
    assert_lease_contract,
    assert_receipt_ingest_contract,
    assert_run_control_contract,
)

_LEASE_UNAVAILABLE_MESSAGE = "Budget lease service temporarily unavailable; retry"


class _ResponseMutationTransport(httpx.BaseTransport):
    def __init__(
        self,
        inner: httpx.BaseTransport,
        mutate: Callable[[httpx.Request, httpx.Response], httpx.Response],
    ) -> None:
        self._inner = inner
        self._mutate = mutate

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        response.read()
        return self._mutate(request, response)

    def close(self) -> None:
        self._inner.close()


def _json_response(
    request: httpx.Request,
    response: httpx.Response,
    payload: dict[str, Any],
) -> httpx.Response:
    return httpx.Response(
        response.status_code,
        json=payload,
        headers=response.headers,
        request=request,
    )


def _raw_json_response(
    request: httpx.Request,
    response: httpx.Response,
    payload: dict[str, Any],
) -> httpx.Response:
    """Serve a body httpx's own JSON encoder refuses (non-finite numbers).

    ``json.loads`` accepts them on the way back in, so this is exactly what a
    server emitting ``Infinity`` would put on the wire.
    """
    return httpx.Response(
        response.status_code,
        content=json.dumps(payload, allow_nan=True).encode(),
        headers={"content-type": "application/json"},
        request=request,
    )


def _check_contract_plane(*, price_hints: dict[str, float] | None = None) -> FakeControlPlane:
    plane = FakeControlPlane(price_hints=price_hints)
    plane.deny_next(period="monthly")
    plane.deny_next(period="run_stopped")
    plane.deny_next(period="tag")
    plane.deny_run("contract-agent-run")
    return plane


def _record_checks(
    served: list[dict[str, Any]],
) -> Callable[[httpx.Request, httpx.Response], httpx.Response]:
    def record(request: httpx.Request, response: httpx.Response) -> httpx.Response:
        if request.url.path == "/api/v1/budgets/check":
            served.append(response.json())
        return response

    return record


def _lease_contract_plane() -> FakeControlPlane:
    plane = FakeControlPlane()
    plane.deny_next(period="monthly", scope="lease")
    return plane


def _run_lease_contract(
    plane: FakeControlPlane,
    *,
    transport: httpx.BaseTransport | None = None,
) -> None:
    with (
        plane.refuse_leases(
            status=409,
            code="lease_holder_cap_exceeded",
            requests=1,
        ),
        httpx.Client(
            transport=plane.transport if transport is None else transport,
            base_url=plane.api_url,
        ) as http,
    ):
        assert_lease_contract(http, plane.api_key)


@pytest.mark.unit
def test_shared_check_contract_against_double() -> None:
    plane = FakeControlPlane(price_hints={"openai": 0.5})
    plane.deny_next(period="monthly")
    plane.deny_next(period="run_stopped")
    plane.deny_next(period="tag")
    plane.deny_run("contract-agent-run")

    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        assert_check_contract(http, plane.api_key)

    # Prove the pack sent the REQUEST payloads it claims to, not just that the
    # double's scripted-verdict queue happened to return the right shape —
    # today the double pops its denial queue ignoring bodies, so this is the
    # only thing in the double lane that would catch a pack payload drift.
    assert plane.checks[0].tags == {"contract_case": "monthly"}
    assert plane.checks[1].agent_run_id == "contract-stopped-run"
    assert plane.checks[2].tags == {"customer": "acme"}
    assert plane.checks[3].agent_run_id == "contract-agent-run"


@pytest.mark.unit
def test_shared_check_contract_opts_into_and_validates_price_hints() -> None:
    # Arrange: a plane that serves hints ONLY to a request carrying the SDK's
    # production price_hints_version opt-in.
    plane = _check_contract_plane(price_hints={"openai": 2.0, "anthropic": 1.0})
    served: list[dict[str, Any]] = []
    transport = _ResponseMutationTransport(plane.transport, _record_checks(served))

    # Act
    with httpx.Client(transport=transport, base_url=plane.api_url) as http:
        assert_check_contract(http, plane.api_key)

    # Assert: the probe sends the SDK's exact wire bytes, so the served hint
    # map is observable — without the opt-in the shape check is unreachable.
    assert [check.price_hints_version for check in plane.checks] == ["1"] * len(plane.checks)
    assert served[0]["price_hints"] == {"openai": 2.0, "anthropic": 1.0}


@pytest.mark.unit
def test_shared_check_contract_rejects_unknown_price_hint_provider() -> None:
    # Arrange: an unknown provider key would fail BudgetCheckResponse
    # validation for the WHOLE response, so the SDK would fail open with no
    # reservation — the probe must name the offending key.
    plane = _check_contract_plane(price_hints={"openai": 2.0})

    def rewrite_hint_provider(
        request: httpx.Request,
        response: httpx.Response,
    ) -> httpx.Response:
        if request.url.path != "/api/v1/budgets/check":
            return response
        payload = response.json()
        payload["price_hints"] = {"brandnew": 1.0}
        return _json_response(request, response, payload)

    transport = _ResponseMutationTransport(plane.transport, rewrite_hint_provider)

    # Act / Assert
    with (
        httpx.Client(transport=transport, base_url=plane.api_url) as http,
        pytest.raises(
            AssertionError,
            match=r"price hint provider 'brandnew' is not a known ProviderName",
        ),
    ):
        assert_check_contract(http, plane.api_key)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("wire_hint", "message"),
    [
        (float("inf"), "must be a finite JSON number"),
        (True, "must be a JSON number"),
    ],
)
def test_shared_check_contract_rejects_unusable_price_hint_values(
    wire_hint: object,
    message: str,
) -> None:
    # Arrange: a hint the SDK's cost policy cannot order by is drift too.
    plane = _check_contract_plane(price_hints={"openai": 2.0})

    def rewrite_hint_value(
        request: httpx.Request,
        response: httpx.Response,
    ) -> httpx.Response:
        if request.url.path != "/api/v1/budgets/check":
            return response
        payload = response.json()
        payload["price_hints"] = {"openai": wire_hint}
        return _raw_json_response(request, response, payload)

    transport = _ResponseMutationTransport(plane.transport, rewrite_hint_value)

    # Act / Assert
    with (
        httpx.Client(transport=transport, base_url=plane.api_url) as http,
        pytest.raises(AssertionError, match=message),
    ):
        assert_check_contract(http, plane.api_key)


@pytest.mark.unit
def test_shared_check_contract_accepts_a_null_price_hint_statement() -> None:
    # Arrange: a plane with no hints. The directive-v1 wire serializes the null
    # statement by omitting the key, exactly as the live API does.
    plane = _check_contract_plane()
    served: list[dict[str, Any]] = []
    transport = _ResponseMutationTransport(plane.transport, _record_checks(served))

    # Act
    with httpx.Client(transport=transport, base_url=plane.api_url) as http:
        assert_check_contract(http, plane.api_key)

    # Assert
    assert served
    assert all("price_hints" not in payload for payload in served)


@pytest.mark.unit
def test_shared_confirm_contract_against_double() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        assert_confirm_contract(http, plane.api_key)


@pytest.mark.unit
def test_shared_run_control_contract_against_double() -> None:
    plane = FakeControlPlane()
    run_id = "contract-stopped-run"
    plane.stop_run(run_id)

    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        assert_run_control_contract(http, plane.api_key, stopped_run_id=run_id)

    assert plane.stopped_runs == {run_id: "manual_kill"}


@pytest.mark.unit
def test_shared_receipt_ingest_contract_against_double() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        assert_receipt_ingest_contract(http, plane.api_key)

    assert [event.deny_source for event in plane.denial_receipts] == [
        "server",
        "aggregate_replay",
    ]
    leased = [event for event in plane.ingested if event.lease_id is not None]
    assert len(leased) == 1
    assert leased[0].status == "success"
    lease_id = leased[0].lease_id
    assert lease_id is not None
    assert lease_id.startswith("lse_")
    assert [event.receipt_aggregate_count for event in plane.aggregate_replays] == [3]


@pytest.mark.unit
def test_shared_lease_contract_against_double() -> None:
    _run_lease_contract(_lease_contract_plane())


@pytest.mark.unit
def test_scripted_lease_unavailable_has_exact_double_contract() -> None:
    plane = FakeControlPlane()
    request = LeaseGrantRequest(
        agent_run_id="contract-lease-unavailable",
        holder_id="contract-unavailable-holder",
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        fail_open=True,
        estimated_input_tokens=1000,
    )

    with (
        plane.refuse_leases(status=503, code="lease_unavailable", requests=1),
        httpx.Client(transport=plane.transport, base_url=plane.api_url) as http,
    ):
        response = http.post(
            "/api/v1/budgets/lease",
            json=request.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {plane.api_key}"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "lease_unavailable",
            "message": _LEASE_UNAVAILABLE_MESSAGE,
        }
    }


@pytest.mark.unit
def test_shared_contract_rejects_renewal_with_incomplete_authority_block() -> None:
    plane = _lease_contract_plane()

    def remove_renewal_field(
        request: httpx.Request,
        response: httpx.Response,
    ) -> httpx.Response:
        if request.url.path != "/api/v1/budgets/lease/renew" or response.status_code != 200:
            return response
        payload = response.json()
        payload.pop("refresh_interval_s")
        return _json_response(request, response, payload)

    transport = _ResponseMutationTransport(plane.transport, remove_renewal_field)
    with pytest.raises(AssertionError, match="lease renewal omitted lease keys"):
        _run_lease_contract(plane, transport=transport)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("target", "wire_boolean", "message"),
    [
        ("generation", True, "eligible grant.generation must be a JSON integer"),
        ("first_surrender", True, "lease surrender.released_tokens must be a JSON integer"),
        ("repeated_surrender", False, "repeated surrender.released_tokens must be a JSON integer"),
    ],
)
def test_shared_contract_rejects_boolean_integer_fields(
    target: str,
    wire_boolean: bool,
    message: str,
) -> None:
    plane = _lease_contract_plane()
    surrender_count = 0

    def replace_integer_with_boolean(
        request: httpx.Request,
        response: httpx.Response,
    ) -> httpx.Response:
        nonlocal surrender_count
        payload = response.json() if response.status_code == 200 else None
        if (
            target == "generation"
            and request.url.path == "/api/v1/budgets/lease"
            and isinstance(payload, dict)
            and payload.get("eligible") is True
            and payload.get("allowed") is True
        ):
            payload["generation"] = wire_boolean
            return _json_response(request, response, payload)
        if request.url.path == "/api/v1/budgets/lease/surrender" and isinstance(payload, dict):
            surrender_count += 1
            expected_count = 1 if target == "first_surrender" else 2
            if surrender_count == expected_count:
                payload["released_tokens"] = wire_boolean
                return _json_response(request, response, payload)
        return response

    transport = _ResponseMutationTransport(plane.transport, replace_integer_with_boolean)
    with pytest.raises(AssertionError, match=message):
        _run_lease_contract(plane, transport=transport)


@pytest.mark.unit
def test_shared_contract_wraps_malformed_json_with_phase_context() -> None:
    plane = FakeControlPlane()
    plane.deny_next(period="monthly")
    plane.deny_next(period="run_stopped")
    plane.deny_next(period="tag")
    plane.deny_run("contract-agent-run")

    def corrupt_first_check(
        request: httpx.Request,
        response: httpx.Response,
    ) -> httpx.Response:
        if request.url.path != "/api/v1/budgets/check":
            return response
        return httpx.Response(response.status_code, content=b"{not-json", request=request)

    transport = _ResponseMutationTransport(plane.transport, corrupt_first_check)
    with (
        httpx.Client(transport=transport, base_url=plane.api_url) as http,
        pytest.raises(AssertionError, match="monthly denial.*malformed JSON"),
    ):
        assert_check_contract(http, plane.api_key)


@pytest.mark.unit
def test_shared_contract_wraps_schema_drift_with_phase_context() -> None:
    plane = FakeControlPlane()
    plane.deny_next(period="monthly")
    plane.deny_next(period="run_stopped")
    plane.deny_next(period="tag")
    plane.deny_run("contract-agent-run")

    def add_unknown_field(
        request: httpx.Request,
        response: httpx.Response,
    ) -> httpx.Response:
        if request.url.path != "/api/v1/budgets/check":
            return response
        payload = response.json()
        payload["contract_schema_drift"] = True
        return _json_response(request, response, payload)

    transport = _ResponseMutationTransport(plane.transport, add_unknown_field)
    with (
        httpx.Client(transport=transport, base_url=plane.api_url) as http,
        pytest.raises(AssertionError, match="monthly denial.*schema validation failed"),
    ):
        assert_check_contract(http, plane.api_key)


@pytest.mark.unit
def test_shared_contract_reports_unexpected_status_with_safe_bounded_preview() -> None:
    plane = FakeControlPlane()
    plane.deny_next(period="monthly")
    plane.deny_next(period="run_stopped")
    plane.deny_next(period="tag")
    plane.deny_run("contract-agent-run")

    def replace_first_check_status(
        request: httpx.Request,
        response: httpx.Response,
    ) -> httpx.Response:
        if request.url.path != "/api/v1/budgets/check":
            return response
        return httpx.Response(
            500,
            json={"detail": "contract-status-preview"},
            request=request,
        )

    transport = _ResponseMutationTransport(plane.transport, replace_first_check_status)
    with (
        httpx.Client(transport=transport, base_url=plane.api_url) as http,
        pytest.raises(AssertionError) as captured,
    ):
        assert_check_contract(http, plane.api_key)

    message = str(captured.value)
    assert "monthly denial" in message
    assert "500" in message
    assert "contract-status-preview" in message
    assert len(message) < 400
    assert plane.api_key not in message
