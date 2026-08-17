"""SDK settlement behavior dogfooded on ``FakeControlPlane``."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from solwyn._token_details import TokenDetails
from solwyn.budget import BudgetEnforcer
from solwyn.reporter import MetadataReporter
from solwyn.testing import FakeControlPlane

_SAMPLE_TOKEN_DETAILS = TokenDetails(input_tokens=100, output_tokens=50)


@contextmanager
def _components(plane: FakeControlPlane) -> Iterator[tuple[BudgetEnforcer, MetadataReporter]]:
    enforcer = BudgetEnforcer(
        plane.api_url,
        plane.api_key,
        cache_ttl=0,
        lease_enabled=False,
        transport=plane.transport,
    )
    reporter = MetadataReporter(
        plane.api_url,
        plane.api_key,
        flush_interval=60,
        max_send_attempts=1,
        transport=plane.transport,
    )
    try:
        yield enforcer, reporter
    finally:
        try:
            reporter.close()
        finally:
            enforcer.close()


@pytest.mark.unit
def test_valid_reservation_builds_records_and_idempotently_replays_confirm() -> None:
    plane = FakeControlPlane()
    with _components(plane) as (enforcer, reporter):
        result = enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-5.5",
            provider="openai",
        )
        assert result.reservation_id is not None
        confirm = enforcer.build_confirm_request(
            reservation_id=result.reservation_id,
            model="gpt-5.5",
            token_details=_SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id=str(uuid.uuid4()),
        )

        reporter._send_confirm(confirm)
        reporter._send_confirm(confirm)

    assert confirm.reservation_id == result.reservation_id
    assert confirm.lease_id is None
    assert plane.confirms == [confirm]
    assert reporter.dropped_counts == {}


@pytest.mark.unit
def test_invalid_reservation_is_best_effort_and_does_not_record_settlement() -> None:
    plane = FakeControlPlane()
    with _components(plane) as (enforcer, reporter):
        confirm = enforcer.build_confirm_request(
            reservation_id="res_nonexistent_000",
            model="gpt-5.5",
            token_details=_SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id=str(uuid.uuid4()),
        )

        reporter._send_confirm(confirm)

    assert plane.confirms == []
    assert reporter._consecutive_confirm_failures == 1


@pytest.mark.unit
def test_queued_invalid_reservation_has_one_counted_terminal_drop() -> None:
    plane = FakeControlPlane()
    with _components(plane) as (enforcer, reporter):
        confirm = enforcer.build_confirm_request(
            reservation_id="res_nonexistent_queued",
            model="gpt-5.5",
            token_details=_SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id=str(uuid.uuid4()),
        )

        reporter.report_confirm(confirm)
        reporter.close(timeout=1.0)

    assert plane.confirms == []
    assert reporter.dropped_counts == {"confirm.terminal_status": 1}
