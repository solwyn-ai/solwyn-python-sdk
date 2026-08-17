"""Shared control-plane contract pack against the in-process double."""

from __future__ import annotations

import httpx
import pytest

from solwyn.testing import FakeControlPlane
from solwyn.testing.contract import (
    assert_check_contract,
    assert_confirm_contract,
    assert_lease_contract,
)


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
    plane = FakeControlPlane()
    plane.deny_next(period="monthly")

    with (
        plane.refuse_leases(status=503, code="lease_unavailable", requests=1),
        plane.refuse_leases(
            status=409,
            code="lease_holder_cap_exceeded",
            requests=1,
        ),
        httpx.Client(transport=plane.transport, base_url=plane.api_url) as http,
    ):
        assert_lease_contract(http, plane.api_key)
