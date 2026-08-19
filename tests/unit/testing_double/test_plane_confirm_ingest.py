"""Confirm, ingest, advisory, and recording behavior of the test double."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import httpx
import pytest

from solwyn._token_details import TokenDetails
from solwyn._types import (
    BreakerStateReport,
    BudgetConfirmRequest,
    CircuitState,
    MetadataEvent,
    ProviderName,
    UntrackedSurfaceReport,
)
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


def _event(*, label: str = "event", attempt_index: int = 0) -> MetadataEvent:
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
        attempt_index=attempt_index,
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
def test_ingest_replays_and_partial_overlap_skip_existing_call_attempt_identities() -> None:
    plane = FakeControlPlane()
    first_batch = [_event(label="dedup-a"), _event(label="dedup-b")]
    overlapping_batch = [_event(label="dedup-b"), _event(label="dedup-c")]

    with httpx.Client(transport=plane.transport) as client:
        first = client.post(
            f"{plane.api_url}/api/v1/metadata/ingest",
            json=[event.model_dump(mode="json") for event in first_batch],
        )
        overlap = client.post(
            f"{plane.api_url}/api/v1/metadata/ingest",
            json=[event.model_dump(mode="json") for event in overlapping_batch],
        )
        replay = client.post(
            f"{plane.api_url}/api/v1/metadata/ingest",
            json=[event.model_dump(mode="json") for event in overlapping_batch],
        )

    assert first.json() == {"ingested": 2, "rejected": []}
    assert overlap.json() == {"ingested": 1, "rejected": []}
    assert replay.json() == {"ingested": 0, "rejected": []}
    assert [event.call_id for event in plane.ingested] == [
        _call_id("dedup-a"),
        _call_id("dedup-b"),
        _call_id("dedup-c"),
    ]


@pytest.mark.unit
def test_ingest_attempt_index_is_part_of_the_production_idempotency_key() -> None:
    plane = FakeControlPlane()
    attempts = [
        _event(label="same-call", attempt_index=0),
        _event(label="same-call", attempt_index=1),
    ]

    response = plane.handle(
        "POST",
        "/api/v1/metadata/ingest",
        [event.model_dump(mode="json") for event in attempts],
    )

    assert response.body == {"ingested": 2, "rejected": []}
    assert plane.ingested == attempts


@pytest.mark.unit
def test_ingest_preserves_core_legacy_timestamp_instance_dedup_surface() -> None:
    plane = FakeControlPlane()
    original = _event(label="legacy-original")
    legacy_collision = original.model_copy(update={"call_id": _call_id("legacy-distinct-call")})

    first = plane.handle(
        "POST",
        "/api/v1/metadata/ingest",
        [original.model_dump(mode="json")],
    )
    replay = plane.handle(
        "POST",
        "/api/v1/metadata/ingest",
        [legacy_collision.model_dump(mode="json")],
    )

    assert first.body == {"ingested": 1, "rejected": []}
    assert replay.body == {"ingested": 0, "rejected": []}
    assert plane.ingested == [original]


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
    # Pin the detail entry structure (index-prefixed loc from parse_model_list's
    # batch wrapping) rather than an isinstance(..., list) shape check, so a
    # mutation that drops/garbles a field still fails this test.
    assert response.json() == {
        "detail": [
            {
                "type": "extra_forbidden",
                "loc": [1, "prompt"],
                "msg": "Extra inputs are not permitted",
            }
        ]
    }
    assert "forbidden contract drift" not in response.text
    assert plane.ingested == []


@pytest.mark.unit
def test_untracked_accepts_up_to_100_and_rejects_larger_batches() -> None:
    plane = FakeControlPlane()
    accepted = [_untracked(index=index) for index in range(2)]
    at_cap = [_untracked(index=index + 200) for index in range(100)]
    too_many = [_untracked(index=index + 400) for index in range(101)]

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[report.model_dump(mode="json") for report in accepted],
        )
        boundary = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[report.model_dump(mode="json") for report in at_cap],
        )
        rejected = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[report.model_dump(mode="json") for report in too_many],
        )

    assert response.status_code == 202
    assert response.json() == {"accepted": 2}
    # Exact boundary: exactly 100 is still accepted...
    assert boundary.status_code == 202
    assert boundary.json() == {"accepted": 100}
    # ...and one more tips it into the 400 batch-size refusal.
    assert rejected.status_code == 400
    assert rejected.json() == {
        "detail": "untracked surface batches may contain at most 100 reports"
    }
    assert plane.untracked_reports == accepted + at_cap


@pytest.mark.unit
def test_untracked_immediate_report_id_replay_is_not_recorded_twice() -> None:
    plane = FakeControlPlane()
    first = _untracked(index=1)
    newer = _untracked(index=2)

    with httpx.Client(transport=plane.transport) as client:
        accepted = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[first.model_dump(mode="json")],
        )
        replay = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[first.model_dump(mode="json")],
        )
        client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[newer.model_dump(mode="json")],
        )
        older_after_newer = client.post(
            f"{plane.api_url}/api/v1/untracked-surfaces",
            json=[first.model_dump(mode="json")],
        )

    assert accepted.json() == {"accepted": 1}
    assert replay.json() == {"accepted": 1}
    assert older_after_newer.json() == {"accepted": 1}
    assert [report.report_id for report in plane.untracked_reports] == [
        first.report_id,
        newer.report_id,
        first.report_id,
    ]


@pytest.mark.unit
def test_reset_recording_preserves_ingest_and_untracked_replay_state() -> None:
    plane = FakeControlPlane()
    event = _event(label="reset-idempotency")
    report = _untracked(index=7)

    plane.handle(
        "POST",
        "/api/v1/metadata/ingest",
        [event.model_dump(mode="json")],
    )
    plane.handle(
        "POST",
        "/api/v1/untracked-surfaces",
        [report.model_dump(mode="json")],
    )
    plane.reset_recording()
    ingest_replay = plane.handle(
        "POST",
        "/api/v1/metadata/ingest",
        [event.model_dump(mode="json")],
    )
    untracked_replay = plane.handle(
        "POST",
        "/api/v1/untracked-surfaces",
        [report.model_dump(mode="json")],
    )

    assert ingest_replay.body == {"ingested": 0, "rejected": []}
    assert untracked_replay.body == {"accepted": 1}
    assert plane.ingested == []
    assert plane.untracked_reports == []


@pytest.mark.unit
def test_breaker_report_is_a_recording_sink_and_health_is_available() -> None:
    plane = FakeControlPlane()
    report = BreakerStateReport(
        provider=ProviderName.OPENAI,
        state=CircuitState.CLOSED,
        failure_count=0,
        success_count=1,
        reported_at=datetime.now(UTC),
        sdk_instance_id="sdk-fake",
    )
    payload = report.model_dump(mode="json")

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/projects/{plane.project_id}/providers/breaker-reports",
            json=payload,
        )
        health = client.get(f"{plane.api_url}/health")

    assert response.status_code == 204
    # Pin every field of the recorded dict against the posted BreakerStateReport
    # — a recording that silently drops a field must fail this, not just a
    # substring check on two of six fields.
    assert plane.breaker_reports == [report.model_dump(mode="json")]
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
async def test_async_ingest_retry_returns_zero_without_duplicate_recording() -> None:
    plane = FakeControlPlane()
    event = _event(label="async-retry")
    payload = [event.model_dump(mode="json")]

    async with httpx.AsyncClient(transport=plane.transport) as client:
        first = await client.post(f"{plane.api_url}/api/v1/metadata/ingest", json=payload)
        replay = await client.post(f"{plane.api_url}/api/v1/metadata/ingest", json=payload)

    assert first.json() == {"ingested": 1, "rejected": []}
    assert replay.json() == {"ingested": 0, "rejected": []}
    assert plane.ingested == [event]


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


@pytest.mark.unit
def test_concurrent_duplicate_ingest_batches_are_claimed_atomically() -> None:
    plane = FakeControlPlane()
    events = [_event(label="concurrent-a"), _event(label="concurrent-b")]
    payload = [event.model_dump(mode="json") for event in events]

    def send(_index: int) -> int:
        with httpx.Client(transport=plane.transport) as client:
            response = client.post(
                f"{plane.api_url}/api/v1/metadata/ingest",
                json=payload,
            )
        return response.json()["ingested"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        ingested_counts = list(executor.map(send, range(40)))

    assert sum(ingested_counts) == 2
    assert plane.ingested == events
