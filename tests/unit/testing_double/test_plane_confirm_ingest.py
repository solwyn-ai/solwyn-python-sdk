"""Confirm, ingest, advisory, and recording behavior of the test double."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import httpx
import pytest

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, UntrackedSurfaceReport
from solwyn.reporter import MetadataReporter
from solwyn.testing import FakeControlPlane


def _call_id(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, label))


def _check_payload(model: str = "gpt-5.5") -> dict[str, object]:
    return {
        "estimated_input_tokens": 3,
        "model": model,
        "provider": "openai",
        "fallback_providers": [],
        "fallback_models": [],
        "failover_directive_version": "1",
    }


def _reservation(client: httpx.Client, plane: FakeControlPlane) -> str:
    response = client.post(
        f"{plane.api_url}/api/v1/budgets/check",
        json=_check_payload(),
    )
    value = response.json()["reservation_id"]
    if not isinstance(value, str):
        raise RuntimeError("test setup expected a reservation id")
    return value


def _confirm(reservation_id: str, *, label: str = "confirm") -> BudgetConfirmRequest:
    return BudgetConfirmRequest(
        reservation_id=reservation_id,
        model="gpt-5.5",
        provider="openai",
        call_id=_call_id(label),
        token_details=TokenDetails(input_tokens=3, output_tokens=2),
    )


def _event(*, label: str = "event") -> MetadataEvent:
    return MetadataEvent(
        model="gpt-5.5",
        provider="openai",
        input_tokens=3,
        output_tokens=2,
        latency_ms=12.5,
        status="success",
        is_model_fallback=False,
        sdk_instance_id="sdk-fake",
        timestamp=datetime.now(UTC),
        call_id=_call_id(label),
    )


def _untracked(*, index: int = 0) -> UntrackedSurfaceReport:
    now = datetime.now(UTC)
    return UntrackedSurfaceReport(
        provider="openai",
        client_shape="openai_sdk",
        mode="sync",
        surface="models.list",
        rule_kind="unmetered_spend",
        capability_scope="operation",
        posture="warn",
        occurrences=1,
        first_seen_at=now,
        last_seen_at=now,
        sdk_instance_id="sdk-fake",
        report_id=_call_id(f"report-{index}"),
    )


@pytest.mark.unit
def test_confirm_is_204_and_deduplicates_call_and_reservation_replays() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        reservation_id = _reservation(client, plane)
        payload = _confirm(reservation_id).model_dump(mode="json")
        first = client.post(
            f"{plane.api_url}/api/v1/budgets/confirm",
            json=payload,
        )
        duplicate_call = client.post(
            f"{plane.api_url}/api/v1/budgets/confirm",
            json=payload,
        )
        reservation_replay = client.post(
            f"{plane.api_url}/api/v1/budgets/confirm",
            json=_confirm(reservation_id, label="reservation-replay").model_dump(mode="json"),
        )

    assert first.status_code == 204
    assert first.content == b""
    assert duplicate_call.status_code == 204
    assert reservation_replay.status_code == 204
    assert plane.confirms == [_confirm(reservation_id)]


@pytest.mark.unit
@pytest.mark.parametrize(
    "settlement_keys",
    [
        {},
        {"reservation_id": "res", "lease_id": "lease"},
    ],
)
def test_confirm_requires_exactly_one_settlement_key(
    settlement_keys: dict[str, str],
) -> None:
    plane = FakeControlPlane()
    payload = _confirm("res").model_dump(mode="json")
    payload.pop("reservation_id")
    payload.update(settlement_keys)

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/budgets/confirm",
            json=payload,
        )

    assert response.status_code == 422
    assert plane.confirms == []


@pytest.mark.unit
def test_expired_reservation_returns_terminal_404_to_real_reporter() -> None:
    plane = FakeControlPlane()
    with httpx.Client(transport=plane.transport) as client:
        reservation_id = _reservation(client, plane)
    plane.expire_reservations()
    reporter = MetadataReporter(
        plane.api_url,
        plane.api_key,
        flush_interval=60,
        max_send_attempts=1,
        transport=plane.transport,
    )

    with httpx.Client(transport=plane.transport) as client:
        expired = client.post(
            f"{plane.api_url}/api/v1/budgets/confirm",
            json=_confirm(reservation_id, label="expired-raw").model_dump(mode="json"),
        )
    reporter.report_confirm(_confirm(reservation_id, label="expired"))
    reporter.close(timeout=1.0)

    assert expired.status_code == 404
    assert expired.json() == {"detail": "Reservation not found or expired"}
    assert reporter.dropped_counts["confirm.terminal_status"] == 1
    assert plane.confirms == []


@pytest.mark.unit
def test_ingest_flattens_validated_events_and_returns_accurate_count() -> None:
    plane = FakeControlPlane()
    events = [_event(label="event-a"), _event(label="event-b")]

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/metadata/ingest",
            json=[event.model_dump(mode="json") for event in events],
        )

    assert response.status_code == 202
    assert response.json() == {"ingested": 2, "rejected": []}
    assert plane.ingested == events


@pytest.mark.unit
def test_invalid_ingest_event_returns_422_without_partial_recording() -> None:
    plane = FakeControlPlane()
    invalid = _event().model_dump(mode="json")
    invalid["prompt"] = "forbidden contract drift"

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/metadata/ingest",
            json=[_event(label="valid").model_dump(mode="json"), invalid],
        )

    assert response.status_code == 422
    assert "forbidden contract drift" not in response.text
    assert plane.ingested == []


@pytest.mark.unit
def test_untracked_accepts_up_to_100_and_rejects_larger_batches() -> None:
    plane = FakeControlPlane()
    accepted = [_untracked(index=index) for index in range(2)]
    too_many = [_untracked(index=index + 100) for index in range(101)]

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[report.model_dump(mode="json") for report in accepted],
        )
        rejected = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[report.model_dump(mode="json") for report in too_many],
        )

    assert response.status_code == 202
    assert response.json() == {"accepted": 2}
    assert rejected.status_code == 400
    assert plane.untracked_reports == accepted


@pytest.mark.unit
def test_breaker_report_is_a_recording_sink_and_health_is_available() -> None:
    plane = FakeControlPlane()
    payload = {
        "provider": "openai",
        "state": "closed",
        "failure_count": 0,
        "success_count": 1,
        "reported_at": datetime.now(UTC).isoformat(),
        "sdk_instance_id": "sdk-fake",
    }

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/projects/{plane.project_id}/providers/breaker-reports",
            json=payload,
        )
        health = client.get(f"{plane.api_url}/health")

    assert response.is_success
    assert len(plane.breaker_reports) == 1
    assert plane.breaker_reports[0]["provider"] == "openai"
    assert plane.breaker_reports[0]["sdk_instance_id"] == "sdk-fake"
    assert health.status_code == 200


@pytest.mark.unit
def test_slow_confirm_does_not_make_reporter_close_overrun_deadline() -> None:
    plane = FakeControlPlane()
    with httpx.Client(transport=plane.transport) as client:
        reservation_id = _reservation(client, plane)
    reporter = MetadataReporter(
        plane.api_url,
        plane.api_key,
        flush_interval=60,
        transport=plane.transport,
    )
    reporter.report_confirm(_confirm(reservation_id, label="slow"))

    started = time.monotonic()
    with plane.slow(0.3, requests=1):
        reporter.close(timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert sum(reporter.dropped_counts.values()) == 1


@pytest.mark.unit
async def test_async_confirm_and_ingest_use_the_same_recording_state() -> None:
    plane = FakeControlPlane()

    async with httpx.AsyncClient(transport=plane.transport) as client:
        check = await client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(),
        )
        reservation_id = check.json()["reservation_id"]
        confirmed = await client.post(
            f"{plane.api_url}/api/v1/budgets/confirm",
            json=_confirm(reservation_id, label="async").model_dump(mode="json"),
        )
        ingested = await client.post(
            f"{plane.api_url}/api/v1/metadata/ingest",
            json=[_event(label="async").model_dump(mode="json")],
        )

    assert confirmed.status_code == 204
    assert ingested.status_code == 202
    assert len(plane.confirms) == 1
    assert len(plane.ingested) == 1


@pytest.mark.unit
def test_recording_is_safe_under_concurrent_check_requests() -> None:
    plane = FakeControlPlane()

    def send(index: int) -> int:
        with httpx.Client(transport=plane.transport) as client:
            return client.post(
                f"{plane.api_url}/api/v1/budgets/check",
                json=_check_payload(model=f"model-{index}"),
            ).status_code

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(send, range(40)))

    assert statuses == [200] * 40
    assert len(plane.checks) == 40
    assert len({check.model for check in plane.checks}) == 40
