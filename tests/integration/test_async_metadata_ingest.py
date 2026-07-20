"""Integration tests for async metadata event ingestion."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from solwyn._types import MetadataEvent, ProviderName
from solwyn.reporter import AsyncMetadataReporter


def _make_event(seq: int = 0) -> MetadataEvent:
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
        call_id=uuid.uuid4().hex,
    )


@pytest.mark.integration
class TestAsyncMetadataReporterDelivery:
    """Async reporter delivers events to the ingest endpoint."""

    @pytest.mark.integration
    async def test_reporter_flushes_on_close(self, test_credentials) -> None:
        """Events queued before close() are flushed without error."""
        reporter = AsyncMetadataReporter(
            api_url=test_credentials.api_url,
            api_key=test_credentials.api_key,
            flush_interval=60.0,  # long interval — force flush via close()
        )
        reporter.start()
        for i in range(5):
            reporter.report(_make_event(seq=i))

        # close() triggers final flush — should not raise
        await reporter.close()

    @pytest.mark.integration
    async def test_reporter_batch_delivery(
        self, async_metadata_reporter: AsyncMetadataReporter
    ) -> None:
        """Multiple events are batched and delivered within flush interval."""
        for i in range(10):
            async_metadata_reporter.report(_make_event(seq=i))

        # Wait for flush interval to fire
        await asyncio.sleep(2.0)

        # No assertion on API side — we verify no exceptions were raised
        # and the reporter is still healthy (queue drained)
        assert len(async_metadata_reporter._queue) < 10

    @pytest.mark.integration
    async def test_reporter_auto_starts_on_first_enqueue(self, test_credentials) -> None:
        """report() auto-starts the flush loop with no start() or close() call."""
        reporter = AsyncMetadataReporter(
            api_url=test_credentials.api_url,
            api_key=test_credentials.api_key,
            flush_interval=1.0,
        )
        for i in range(5):
            reporter.report(_make_event(seq=i))

        # First enqueue happened inside a running loop -- the flush task must
        # have auto-started, with neither start() nor close() called above.
        assert reporter._flush_task is not None
        assert not reporter._flush_task.done()

        # Delivery happens on the auto-started loop's own cadence.
        await asyncio.sleep(2.0)
        assert len(reporter._queue) == 0

        await reporter.close()

    @pytest.mark.integration
    async def test_reporter_close_without_start(self, test_credentials) -> None:
        """close() on a never-started, empty reporter does not raise."""
        reporter = AsyncMetadataReporter(
            api_url=test_credentials.api_url,
            api_key=test_credentials.api_key,
        )

        # Never enqueued, never started -- close() must still shut down cleanly.
        await reporter.close()
