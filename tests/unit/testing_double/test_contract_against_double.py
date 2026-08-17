"""Shared control-plane contract pack against the in-process double."""

from __future__ import annotations

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
    plane = FakeControlPlane()
    plane.deny_next(period="monthly")
    plane.deny_next(period="run_stopped")
    plane.deny_next(period="tag")
    plane.deny_run("contract-agent-run")

    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        assert_check_contract(http, plane.api_key)


@pytest.mark.unit
def test_shared_confirm_contract_against_double() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        assert_confirm_contract(http, plane.api_key)


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
