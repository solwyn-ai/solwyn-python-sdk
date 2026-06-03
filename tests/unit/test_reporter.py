"""Tests for metadata reporter."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY
from pydantic import ValidationError

from solwyn._token_details import TokenDetails
from solwyn._types import (
    SERVICE_TIER_MAX_LENGTH,
    BudgetConfirmRequest,
    MetadataEvent,
    ProviderName,
)
from solwyn.reporter import (
    AsyncMetadataReporter,
    MetadataReporter,
    _ReporterBase,
)


def _make_event(**overrides) -> MetadataEvent:
    """Create a MetadataEvent with sensible test defaults."""
    defaults = {
        "model": "gpt-4o",
        "provider": ProviderName.OPENAI,
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 200.0,
        "status": "success",
        "is_model_fallback": False,
        "sdk_instance_id": "test-instance-001",
        "timestamp": datetime.now(UTC),
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


def _make_confirm_request() -> BudgetConfirmRequest:
    return BudgetConfirmRequest(
        reservation_id="res_123",
        model="gpt-4o",
        provider=ProviderName.OPENAI,
        call_id="call_sync_confirm",
        token_details=TokenDetails(input_tokens=10, output_tokens=5),
    )


def _error_response(status_code: int) -> MagicMock:
    """A 4xx/5xx httpx.Response stand-in: raise_for_status raises HTTPStatusError."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "error", request=MagicMock(spec=httpx.Request), response=resp
        )
    )
    return resp


def _quiet_sync_reporter(**kwargs) -> MetadataReporter:
    """Build a sync reporter with its background flush thread stopped."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            **kwargs,
        )
    reporter._shutdown.set()
    reporter._thread.join(timeout=2.0)
    return reporter


# ---------------------------------------------------------------------------
# Base class (sans-I/O) tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReporterBase:
    """Tests for _ReporterBase sans-I/O logic."""

    def test_enqueue_adds_event(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        event = _make_event()
        base._enqueue(event)
        assert len(base._queue) == 1

    def test_drain_batch_returns_up_to_batch_size(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            batch_size=3,
        )
        for _ in range(5):
            base._enqueue(_make_event())

        batch = base._drain_batch()
        assert len(batch) == 3
        assert len(base._queue) == 2

    def test_drain_batch_returns_all_when_less_than_batch_size(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            batch_size=10,
        )
        for _ in range(3):
            base._enqueue(_make_event())

        batch = base._drain_batch()
        assert len(batch) == 3
        assert len(base._queue) == 0

    def test_queue_overflow_drops_oldest(self) -> None:
        base = _ReporterBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            max_queue_size=3,
        )

        events = []
        for i in range(5):
            event = _make_event(input_tokens=i)
            events.append(event)
            base._enqueue(event)

        # Queue should contain only the 3 most recent
        assert len(base._queue) == 3
        assert base._queue[0].input_tokens == 2
        assert base._queue[1].input_tokens == 3
        assert base._queue[2].input_tokens == 4


# ---------------------------------------------------------------------------
# Sync reporter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetadataReporter:
    """Tests for the synchronous MetadataReporter."""

    def test_report_enqueues_event(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            )
            # Stop the background thread to avoid timing issues
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            event = _make_event()
            reporter.report(event)
            assert len(reporter._queue) == 1
            reporter._http.close()

    def test_batch_flush_triggers_at_batch_size(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
                batch_size=3,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            # Enqueue 5 events
            for _ in range(5):
                reporter.report(_make_event())

            # Mock the HTTP call
            mock_response = MagicMock()
            with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
                reporter._flush_remaining()

            # Should have sent 2 batches (3 + 2)
            assert mock_post.call_count == 2
            reporter._http.close()

    def test_graceful_shutdown_flushes_remaining(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
                batch_size=10,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            for _ in range(3):
                reporter.report(_make_event())

            assert len(reporter._queue) == 3

            mock_response = MagicMock()
            with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
                reporter.close()

            # Should have flushed all remaining events
            assert mock_post.call_count == 1
            assert len(reporter._queue) == 0

    def test_context_manager(self) -> None:
        with (
            patch("solwyn.reporter.MetadataReporter._flush_loop"),
            MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            ) as reporter,
        ):
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)
            reporter.report(_make_event())
        # After context exit, reporter should be closed

    def test_send_batch_omits_none_fields_from_payload(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            with patch.object(reporter._http, "post") as mock_post:
                reporter._send_batch([_make_event(service_tier=None)])

            payload = mock_post.call_args.kwargs["json"][0]
            assert "service_tier" not in payload
            assert "token_details" not in payload
            reporter._http.close()

    def test_send_batch_includes_service_tier_when_present(self) -> None:
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            with patch.object(reporter._http, "post") as mock_post:
                reporter._send_batch([_make_event(service_tier="priority")])

            payload = mock_post.call_args.kwargs["json"][0]
            assert payload["service_tier"] == "priority"
            reporter._http.close()

    def test_send_batch_includes_agent_run_fields_when_set(self) -> None:
        # Verify the new agent_run_* fields reach the wire payload when set
        # on the event. Guards against regressions in the model_dump path.
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            event = _make_event(
                agent_run_id="run_test_abc",
                agent_run_name="test-batch",
            )
            with patch.object(reporter._http, "post") as mock_post:
                reporter._send_batch([event])

            payload = mock_post.call_args.kwargs["json"][0]
            assert payload["agent_run_id"] == "run_test_abc"
            assert payload["agent_run_name"] == "test-batch"
            reporter._http.close()

    def test_send_batch_omits_agent_run_fields_when_none(self) -> None:
        # When no run scope is active, the wire payload must omit the
        # keys entirely so the API's per-day fallback id engages.
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            reporter = MetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
            )
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)

            with patch.object(reporter._http, "post") as mock_post:
                reporter._send_batch([_make_event()])

            payload = mock_post.call_args.kwargs["json"][0]
            assert "agent_run_id" not in payload
            assert "agent_run_name" not in payload
            reporter._http.close()

    def test_metadata_event_rejects_overlength_service_tier(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(service_tier="x" * (SERVICE_TIER_MAX_LENGTH + 1))

    def test_send_batch_4xx_logged_type_only_and_survives(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A 4xx on the ingest POST must surface via raise_for_status, be caught,
        # logged by class name only (no body), and not propagate.
        reporter = _quiet_sync_reporter()

        with (
            patch.object(reporter._http, "post", return_value=_error_response(422)),
            caplog.at_level("WARNING"),
        ):
            # Should not raise despite the 422.
            reporter._send_batch([_make_event()])

        assert "HTTPStatusError" in caplog.text
        # Privacy: never log the response body — only the exception class name.
        assert "422 Unprocessable" not in caplog.text
        reporter._http.close()

    def test_flush_remaining_confirm_4xx_logged_type_only_and_survives(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A 4xx on the confirm POST must surface via raise_for_status, be caught,
        # logged by class name only, drain from the queue, and not propagate.
        reporter = _quiet_sync_reporter()
        # Enqueue directly: report_confirm gates on the shutdown flag that
        # _quiet_sync_reporter sets to stop the background thread.
        reporter._confirm_queue.append(_make_confirm_request())

        with (
            patch.object(reporter._http, "post", return_value=_error_response(422)),
            caplog.at_level("WARNING"),
        ):
            # Should not raise despite the 422.
            reporter._flush_remaining()

        assert "reporter.confirm_send_failed" in caplog.text
        assert "HTTPStatusError" in caplog.text
        # The flush loop survives and drains the confirm queue.
        assert len(reporter._confirm_queue) == 0
        reporter._http.close()


# ---------------------------------------------------------------------------
# Async reporter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncMetadataReporter:
    """Tests for the async reporter base logic (no event loop needed)."""

    def test_report_enqueues_event(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        event = _make_event()
        reporter.report(event)
        assert len(reporter._queue) == 1

    def test_queue_overflow(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            max_queue_size=2,
        )
        reporter.report(_make_event(input_tokens=1))
        reporter.report(_make_event(input_tokens=2))
        reporter.report(_make_event(input_tokens=3))

        assert len(reporter._queue) == 2
        assert reporter._queue[0].input_tokens == 2
        assert reporter._queue[1].input_tokens == 3
