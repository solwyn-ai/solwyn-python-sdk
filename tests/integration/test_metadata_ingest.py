"""Integration tests for metadata event ingestion."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

import pytest

from solwyn._types import MetadataEvent, ProviderName
from solwyn.reporter import MetadataReporter


def _make_event(seq: int = 0) -> MetadataEvent:
    """Create a minimal metadata event for testing."""
    return MetadataEvent(
        model="gpt-4o",
        provider=ProviderName.OPENAI,
        input_tokens=100 + seq,
        output_tokens=50 + seq,
        latency_ms=150.0,
        status="success",
        is_model_fallback=False,
        sdk_instance_id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        call_id=uuid.uuid4().hex,
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
