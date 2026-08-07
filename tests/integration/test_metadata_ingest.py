"""Integration tests for metadata event ingestion."""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime

import httpx
import pytest

from solwyn._types import MetadataEvent, ProviderName
from solwyn.reporter import MetadataReporter

_TAGS_API_KEY_ENV = "SOLWYN_TEST_TAGS_API_KEY"
_ROUND_TRIP_TAG_KEY = "sdk_t20_live_roundtrip"
_ROUND_TRIP_TAG_VALUE = "tagged"


@pytest.fixture(scope="session")
def paid_tags_api_key() -> str:
    """Require an explicitly provisioned Team/Scale key for tag readback."""
    api_key = os.environ.get(_TAGS_API_KEY_ENV)
    if not api_key:
        pytest.skip(
            f"live tag round-trip requires a dedicated Team/Scale project key in "
            f"{_TAGS_API_KEY_ENV}"
        )
    return api_key


def _make_event(
    seq: int = 0,
    *,
    tags: dict[str, str] | None = None,
    agent_run_id: str | None = None,
) -> MetadataEvent:
    """Create a minimal metadata event for testing."""
    return MetadataEvent(
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        input_tokens=100 + seq,
        output_tokens=50 + seq,
        latency_ms=150.0,
        status="success",
        is_model_fallback=False,
        sdk_instance_id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        call_id=str(uuid.uuid4()),
        tags=tags,
        agent_run_id=agent_run_id,
    )


@pytest.mark.integration
class TestMetadataReporterDelivery:
    """Reporter delivers events to the ingest endpoint."""

    @pytest.mark.integration
    def test_reporter_flushes_on_close(self, test_credentials) -> None:
        """Events queued before close() are flushed without error."""
        reporter = MetadataReporter(
            api_url=test_credentials.api_url,
            api_key=test_credentials.api_key,
            flush_interval=60.0,  # long interval — force flush via close()
        )
        for i in range(5):
            reporter.report(_make_event(seq=i))

        # close() triggers final flush — should not raise
        reporter.close()

    @pytest.mark.integration
    def test_reporter_batch_delivery(self, metadata_reporter: MetadataReporter) -> None:
        """Multiple events are batched and delivered within flush interval."""
        for i in range(10):
            metadata_reporter.report(_make_event(seq=i))

        # Wait for flush interval to fire
        time.sleep(2.0)

        # No assertion on API side — we verify no exceptions were raised
        # and the reporter is still healthy (queue drained)
        assert len(metadata_reporter._queue) < 10

    def test_tags_round_trip_into_cost_grouping(
        self,
        api_url: str,
        paid_tags_api_key: str,
    ) -> None:
        agent_run_id = f"sdk-t20-{uuid.uuid4().hex}"
        event = _make_event(
            tags={_ROUND_TRIP_TAG_KEY: _ROUND_TRIP_TAG_VALUE},
            agent_run_id=agent_run_id,
        )
        headers = {"Authorization": f"Bearer {paid_tags_api_key}"}

        with httpx.Client(base_url=api_url, timeout=15) as http:
            project_response = http.post(
                "/api/v1/budgets/check",
                json={
                    "estimated_input_tokens": 1,
                    "model": event.model,
                    "provider": event.provider.value,
                },
                headers=headers,
            )
            project_response.raise_for_status()
            project_id = project_response.json()["project_id"]

            costs_path = f"/api/v1/projects/{project_id}/costs"
            costs_params = {
                "range": "7d",
                "group_by": f"tag:{_ROUND_TRIP_TAG_KEY}",
                "agent_run": agent_run_id,
            }
            entitlement_response = http.get(
                costs_path,
                params=costs_params,
                headers=headers,
            )
            if entitlement_response.status_code == 402:
                pytest.fail(
                    f"{_TAGS_API_KEY_ENV} must contain a Team/Scale project key "
                    "with tag grouping enabled"
                )
            entitlement_response.raise_for_status()

            ingest_response = http.post(
                "/api/v1/metadata/ingest",
                json=[event.model_dump(mode="json", exclude_none=True)],
                headers=headers,
            )
            ingest_response.raise_for_status()
            assert ingest_response.status_code == 202
            assert ingest_response.json()["ingested"] == 1
            assert ingest_response.json()["rejected"] == []

            costs_response = http.get(
                costs_path,
                params=costs_params,
                headers=headers,
            )
            costs_response.raise_for_status()

        body = costs_response.json()
        assert body["group_by"] == f"tag:{_ROUND_TRIP_TAG_KEY}"
        matching_groups = [
            group for group in body["groups"] if group["group_key"] == _ROUND_TRIP_TAG_VALUE
        ]
        assert len(matching_groups) == 1
        assert matching_groups[0]["call_count"] == 1
