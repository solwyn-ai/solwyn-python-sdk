"""Denial-receipt recording and scriptable ingest rejections on the double."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from solwyn._types import MetadataEvent
from solwyn.reporter import MetadataReporter
from solwyn.testing import FakeControlPlane

_INGEST_PATH = "/api/v1/metadata/ingest"


def _receipt_event(**overrides: object) -> MetadataEvent:
    """Build one valid denied-call receipt with a fresh call id."""
    values: dict[str, Any] = {
        "model": "gpt-5.5",
        "provider": "openai",
        "input_tokens": 3,
        "output_tokens": 0,
        "latency_ms": 0.0,
        "status": "budget_denied",
        "is_model_fallback": False,
        "sdk_instance_id": "sdk-receipt",
        "timestamp": datetime.now(UTC),
        "call_id": str(uuid.uuid4()),
        "agent_run_id": "run-receipt",
        "deny_source": "server",
        "deny_reason": "manual_kill",
        "denied_by_period": "run_stopped",
        "estimated_output_bound": 4096,
        "velocity_flags": [],
    }
    values.update(overrides)
    return MetadataEvent(**values)


def _receipt_payload(**overrides: object) -> dict[str, Any]:
    """Build a receipt JSON body, including shapes the wire model rejects."""
    payload = _receipt_event().model_dump(mode="json")
    payload.update(overrides)
    return payload


def _ordinary_event() -> MetadataEvent:
    """Build a plain success event: telemetry that carries no receipt."""
    return _receipt_event(
        status="success",
        output_tokens=2,
        deny_source=None,
        deny_reason=None,
        denied_by_period=None,
        estimated_output_bound=None,
        velocity_flags=None,
    )


def _ingest(plane: FakeControlPlane, events: list[MetadataEvent]) -> httpx.Response:
    with httpx.Client(transport=plane.transport) as client:
        return client.post(
            f"{plane.api_url}{_INGEST_PATH}",
            json=[event.model_dump(mode="json") for event in events],
        )


def _wait_until(
    predicate: Any,
    *,
    timeout: float = 30.0,
    interval: float = 0.005,
    nudge: Any = None,
) -> None:
    # Generous: the reporter's own flush thread paces the cycles, so this
    # returns the moment the condition holds and only a genuinely stalled
    # reporter ever spends the deadline. ``nudge`` runs once per poll for
    # conditions the reporter reaches only when fed more work.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if nudge is not None:
            nudge()
        time.sleep(interval)
    raise AssertionError("condition was not reached before the deadline")


@pytest.mark.unit
def test_denial_receipt_is_accepted_and_recorded() -> None:
    plane = FakeControlPlane()

    response = _ingest(plane, [_receipt_event(), _ordinary_event()])

    assert response.status_code == 202
    assert response.json() == {"ingested": 2, "rejected": []}
    (receipt,) = plane.denial_receipts
    assert receipt.deny_source == "server"
    assert receipt.deny_reason == "manual_kill"
    assert receipt.denied_by_period == "run_stopped"
    assert receipt.estimated_output_bound == 4096
    assert plane.aggregate_replays == []
    assert len(plane.ingested) == 2


@pytest.mark.unit
def test_aggregate_replay_receipt_with_null_pricing_basis_is_accepted() -> None:
    plane = FakeControlPlane()
    # COARSE aggregate: represents many receipts with no single pricing basis,
    # so the receipt_pricing_input_tokens key is absent from the wire entirely.
    event = _receipt_event(
        deny_source="aggregate_replay",
        receipt_aggregate_count=17,
        input_tokens=123_456,
    )

    response = _ingest(plane, [event])

    assert response.status_code == 202
    (replay,) = plane.aggregate_replays
    assert replay.receipt_aggregate_count == 17
    assert replay.receipt_pricing_input_tokens is None
    assert replay.input_tokens == 123_456
    assert plane.denial_receipts == [replay]


@pytest.mark.unit
def test_pricing_basis_without_aggregate_count_is_rejected() -> None:
    plane = FakeControlPlane()
    payload = _receipt_payload(receipt_pricing_input_tokens=500)

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(f"{plane.api_url}{_INGEST_PATH}", json=[payload])

    # The vendored model's own validator, mirrored server-side: the whole
    # request fails at parse time rather than yielding a rejection entry.
    assert response.status_code == 422
    assert plane.ingested == []
    assert plane.denial_receipts == []


@pytest.mark.unit
def test_velocity_flags_beyond_the_wire_bound_are_rejected() -> None:
    plane = FakeControlPlane()
    payload = _receipt_payload(velocity_flags=["repeat_size"] * 9)

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(f"{plane.api_url}{_INGEST_PATH}", json=[payload])

    assert response.status_code == 422
    assert plane.ingested == []


@pytest.mark.unit
def test_reject_ingest_exact_mode_names_indices() -> None:
    plane = FakeControlPlane()
    batch = [_receipt_event(), _receipt_event(model="claude-4.7"), _receipt_event()]

    with plane.reject_ingest(indices=[1], code="unknown_model"):
        body = _ingest(plane, batch).json()

    assert body["ingested"] == 2
    (entry,) = body["rejected"]
    assert entry["index"] == 1
    assert entry["code"] == "unknown_model"
    assert entry["model"] == "claude-4.7"
    assert set(entry) == {"index", "code", "model", "message"}
    assert isinstance(entry["message"], str) and entry["message"]


@pytest.mark.unit
def test_reject_ingest_legacy_mode_omits_index() -> None:
    plane = FakeControlPlane()

    with plane.reject_ingest(count=1):
        body = _ingest(plane, [_receipt_event(), _receipt_event()]).json()

    assert body["ingested"] == 1
    (entry,) = body["rejected"]
    assert "index" not in entry
    assert set(entry) == {"code", "model", "message"}
    assert entry["code"] == "invalid_tags"


@pytest.mark.unit
def test_reject_ingest_malformed_mode_returns_a_non_list_rejected_field() -> None:
    plane = FakeControlPlane()

    with plane.reject_ingest(malformed=True):
        response = _ingest(plane, [_receipt_event()])

    assert response.status_code == 202
    assert response.json() == {"rejected": "corrupt"}


@pytest.mark.unit
def test_rejected_events_are_still_recorded_by_the_plane() -> None:
    plane = FakeControlPlane()
    batch = [_receipt_event(), _receipt_event()]

    with plane.reject_ingest(indices=[0, 1]):
        body = _ingest(plane, batch).json()

    assert body["ingested"] == 0
    assert [entry["index"] for entry in body["rejected"]] == [0, 1]
    assert plane.ingested == batch
    assert plane.denial_receipts == batch


@pytest.mark.unit
def test_reject_ingest_window_is_request_bounded_and_scoped_to_ingest() -> None:
    plane = FakeControlPlane()

    with plane.reject_ingest(count=1, requests=1):
        with httpx.Client(transport=plane.transport) as client:
            # Background control-plane traffic must not spend the script.
            client.post(
                f"{plane.api_url}/api/v1/budgets/check",
                json={
                    "estimated_input_tokens": 3,
                    "model": "gpt-5.5",
                    "provider": "openai",
                    "fallback_providers": [],
                    "fallback_models": [],
                },
            )
        first = _ingest(plane, [_receipt_event()]).json()
        second = _ingest(plane, [_receipt_event()]).json()

    assert len(first["rejected"]) == 1
    assert second == {"ingested": 1, "rejected": []}


@pytest.mark.unit
def test_reject_ingest_requires_exactly_one_mode() -> None:
    plane = FakeControlPlane()

    with pytest.raises(RuntimeError, match="exactly one"), plane.reject_ingest():
        pass

    with (
        pytest.raises(RuntimeError, match="exactly one"),
        plane.reject_ingest(indices=[0], count=1),
    ):
        pass


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"indices": []}, "must not be empty"),
        ({"indices": [-1]}, "must not be negative"),
        ({"indices": [0, 0]}, "must not repeat"),
        ({"count": 0}, "at least one"),
        ({"indices": [0], "code": "not-a-code"}, "code must be one of"),
    ],
)
def test_reject_ingest_rejects_unscriptable_arguments(kwargs: dict[str, Any], match: str) -> None:
    plane = FakeControlPlane()

    with pytest.raises(ValueError, match=match), plane.reject_ingest(**kwargs):
        pass


@pytest.mark.unit
def test_reject_ingest_index_outside_the_batch_fails_loud() -> None:
    plane = FakeControlPlane()

    with pytest.raises(RuntimeError, match="outside"), plane.reject_ingest(indices=[3]):
        _ingest(plane, [_receipt_event()])


@pytest.mark.unit
def test_denial_receipt_views_follow_reset_recording() -> None:
    plane = FakeControlPlane()
    _ingest(plane, [_receipt_event()])
    assert len(plane.denial_receipts) == 1

    plane.reset_recording()

    assert plane.denial_receipts == []
    assert plane.aggregate_replays == []


@pytest.mark.unit
def test_rejected_receipt_folds_and_replays_after_a_clean_cycle() -> None:
    plane = FakeControlPlane()
    reporter = MetadataReporter(
        plane.api_url,
        plane.api_key,
        flush_interval=0.01,
        transport=plane.transport,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    denied = _receipt_event(input_tokens=11, estimated_output_bound=7)
    try:
        with plane.reject_ingest(indices=[0], requests=1):
            reporter.report(denied)
            # denial_receipts takes the plane lock, so the window can only be
            # torn down once the rejecting response is fully built.
            _wait_until(lambda: plane.denial_receipts)

        # A rejected batch is no recovery proof; a clean cycle is, and the
        # cycle AFTER a clean one replays the fold. Offer ordinary events
        # until the replay lands: an event that happens to ride the same
        # cycle as the rejected batch proves nothing, so one offer is not
        # guaranteed to be the clean cycle.
        _wait_until(
            lambda: plane.aggregate_replays,
            nudge=lambda: reporter.report(_ordinary_event()),
            interval=0.05,
        )
    finally:
        reporter.close()

    (replay,) = plane.aggregate_replays
    assert replay.deny_source == "aggregate_replay"
    assert (replay.receipt_aggregate_count or 0) >= 1
    assert replay.call_id != denied.call_id
    assert replay.deny_reason == denied.deny_reason
    assert replay.denied_by_period == denied.denied_by_period
    assert replay.agent_run_id == denied.agent_run_id
    assert replay.input_tokens == denied.input_tokens
    assert replay.estimated_output_bound == denied.estimated_output_bound
