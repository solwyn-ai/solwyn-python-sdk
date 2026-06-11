"""Tests for metadata reporter."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, _accepted_response
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
        "call_id": "call_reporter_event",
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


def _make_confirm_request(**overrides) -> BudgetConfirmRequest:
    defaults = {
        "reservation_id": "res_123",
        "model": "gpt-4o",
        "provider": ProviderName.OPENAI,
        "call_id": "call_sync_confirm",
        "token_details": TokenDetails(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return BudgetConfirmRequest(**defaults)


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


def _rejection(
    index: int,
    *,
    code: str = "unknown_model",
    model: str = "vendor-x-1",
    message: str = "Solwyn does not have pricing for this model. File an issue.",
) -> dict[str, Any]:
    """One IngestRejection wire entry, as the server sends it."""
    return {"index": index, "code": code, "model": model, "message": message}


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


def _unstarted_sync_reporter(**kwargs) -> MetadataReporter:
    """Build a sync reporter whose background thread has already exited."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            **kwargs,
        )
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

    def test_flush_remaining_sends_queued_confirms_before_success_events(self) -> None:
        reporter = _quiet_sync_reporter()
        call_id = "call_stream_settlement"
        reporter.report(_make_event(call_id=call_id))
        # Enqueue directly: report_confirm gates on the shutdown flag that
        # _quiet_sync_reporter sets to stop the background thread.
        reporter._confirm_queue.append(_make_confirm_request(call_id=call_id))

        with patch.object(reporter._http, "post", return_value=MagicMock()) as mock_post:
            reporter._flush_remaining()

        assert [call.args[0] for call in mock_post.call_args_list] == [
            "https://api.test.solwyn.ai/api/v1/budgets/confirm",
            "https://api.test.solwyn.ai/api/v1/metadata/ingest",
        ]
        assert mock_post.call_args_list[0].kwargs["json"]["call_id"] == call_id
        assert mock_post.call_args_list[1].kwargs["json"][0]["call_id"] == call_id
        assert len(reporter._queue) == 0
        assert len(reporter._confirm_queue) == 0
        reporter._http.close()

    def test_settlement_enqueued_during_metadata_send_waits_for_confirm_first(self) -> None:
        reporter = _unstarted_sync_reporter(batch_size=1)
        older_call_id = "call_older_metadata"
        settlement_call_id = "call_mid_flush_settlement"
        reporter.report(_make_event(call_id=older_call_id))
        settlement_confirm = _make_confirm_request(call_id=settlement_call_id)
        settlement_event = _make_event(call_id=settlement_call_id)
        calls: list[tuple[str, Any]] = []
        did_enqueue = False

        def post(url: str, **kwargs: Any) -> MagicMock:
            nonlocal did_enqueue
            payload = kwargs["json"]
            calls.append((url, payload))
            if (
                not did_enqueue
                and "metadata/ingest" in url
                and payload[0]["call_id"] == older_call_id
            ):
                did_enqueue = True
                reporter.report_settlement(settlement_confirm, settlement_event)
            return MagicMock()

        with patch.object(reporter._http, "post", side_effect=post):
            reporter._flush_remaining()
            reporter._flush_remaining()

        settlement_posts = []
        for url, payload in calls:
            if "budgets/confirm" in url and payload["call_id"] == settlement_call_id:
                settlement_posts.append("confirm")
            elif "metadata/ingest" in url and payload[0]["call_id"] == settlement_call_id:
                settlement_posts.append("metadata")

        assert settlement_posts == ["confirm", "metadata"]
        reporter._http.close()

    def test_flush_remaining_confirm_persistent_failures_escalate_to_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reporter = _quiet_sync_reporter()
        for _ in range(10):
            reporter._confirm_queue.append(_make_confirm_request())

        with (
            patch.object(reporter._http, "post", return_value=_error_response(503)),
            caplog.at_level("ERROR"),
        ):
            reporter._flush_remaining()

        assert reporter._consecutive_confirm_failures == 10
        assert "reporter.confirm_send_persistent_failure" in caplog.text
        assert "consecutive_failures=10" in caplog.text
        assert "HTTPStatusError" in caplog.text
        reporter._http.close()


# ---------------------------------------------------------------------------
# Per-event ingest rejection logging (v0.1.7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIngestRejectionLogging:
    """Per-event rejection dispositions in the 202 ingest body.

    The server returns 202 for every well-formed request and lists rejected
    events in the body. Rejected events are terminal for that flush — they
    reject identically on every resubmission until a pricing entry lands
    server-side — so the SDK logs them (aggregated) and drops them.
    """

    def test_mixed_batch_logs_one_warning_per_distinct_code_and_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 5 submitted, 2 rejected on one (code, model) and 1 on another:
        # exactly TWO aggregated lines, never one line per rejected event.
        reporter = _quiet_sync_reporter()
        body = {
            "ingested": 2,
            "rejected": [
                _rejection(0, code="unknown_model", model="vendor-x-1"),
                _rejection(2, code="unknown_model", model="vendor-x-1"),
                _rejection(4, code="unknown_service_tier", model="gpt-4o"),
            ],
        }

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event() for _ in range(5)])

        rejection_logs = [
            record.getMessage()
            for record in caplog.records
            if "reporter.ingest_events_rejected" in record.getMessage()
            and record.levelname == "WARNING"
        ]
        assert len(rejection_logs) == 2
        assert any("code=unknown_model model=vendor-x-1 count=2" in line for line in rejection_logs)
        assert any(
            "code=unknown_service_tier model=gpt-4o count=1" in line for line in rejection_logs
        )
        reporter._http.close()

    def test_all_rejected_batch_logs_and_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # ingested=0 is still a 202 — the flush loop must survive it.
        reporter = _quiet_sync_reporter()
        body = {"ingested": 0, "rejected": [_rejection(i) for i in range(3)]}

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event() for _ in range(3)])

        assert "code=unknown_model model=vendor-x-1 count=3" in caplog.text
        reporter._http.close()

    def test_empty_rejected_list_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        # The server always sends rejected (possibly []) — the clean-batch
        # path must stay silent, exactly like pre-v0.1.7.
        reporter = _quiet_sync_reporter()
        body = {"ingested": 3, "rejected": []}

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event() for _ in range(3)])

        assert not caplog.records
        reporter._http.close()

    def test_rejected_events_are_not_requeued(self) -> None:
        # Rejected events are terminal for the flush: the queue stays drained
        # and a second flush has nothing to send.
        reporter = _quiet_sync_reporter()
        reporter.report(_make_event())
        reporter.report(_make_event())
        body = {"ingested": 1, "rejected": [_rejection(0)]}

        with patch.object(reporter._http, "post", return_value=_accepted_response(body)):
            reporter._flush_remaining()

        assert len(reporter._queue) == 0

        with patch.object(reporter._http, "post") as mock_post:
            reporter._flush_remaining()
        assert mock_post.call_count == 0
        reporter._http.close()

    def test_rejection_message_logged_verbatim(self, caplog: pytest.LogCaptureFixture) -> None:
        # The server repr-escapes its message; the SDK logs it verbatim and
        # never parses or re-interprets it — including %-format specifiers
        # and escaped control characters.
        reporter = _quiet_sync_reporter()
        message = r"service_tier 'spot\x1b[31m\n' is unknown; 100% of cases resolve in %s %d"
        body = {
            "ingested": 0,
            "rejected": [_rejection(0, code="unknown_service_tier", message=message)],
        }

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event()])

        assert message in caplog.text
        reporter._http.close()

    def test_unknown_future_rejection_code_logged_without_crash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The code enum is server-owned and may grow; an unrecognized code is
        # logged like any other, never special-cased and never a crash.
        reporter = _quiet_sync_reporter()
        body = {"ingested": 0, "rejected": [_rejection(0, code="unknown_pricing_dimension")]}

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event()])

        assert "code=unknown_pricing_dimension" in caplog.text
        reporter._http.close()

    @pytest.mark.parametrize(
        ("body", "expected_exc_type"),
        [
            ({"ingested": 3}, "KeyError"),  # rejected key missing
            ({"ingested": 1, "rejected": None}, "TypeError"),  # falsy non-list is not clean
            ({"ingested": 1, "rejected": ["not-a-dict"]}, "TypeError"),
            ({"ingested": 1, "rejected": [{"index": 0}]}, "KeyError"),  # entry missing fields
        ],
    )
    def test_malformed_202_body_fails_open_and_logs_once(
        self,
        body: dict[str, Any],
        expected_exc_type: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Fail-open: a contract-violating body must never raise into the
        # flush loop — fall back to count-only acknowledgment, one log line.
        reporter = _quiet_sync_reporter()

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event()])

        unparseable = [
            record.getMessage()
            for record in caplog.records
            if "reporter.ingest_response_unparseable" in record.getMessage()
        ]
        assert len(unparseable) == 1
        assert f"exc_type={expected_exc_type}" in unparseable[0]
        reporter._http.close()

    def test_invalid_json_202_body_fails_open(self, caplog: pytest.LogCaptureFixture) -> None:
        # A body that is not JSON at all (resp.json() raises) takes the same
        # fail-open path.
        reporter = _quiet_sync_reporter()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 202
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(side_effect=ValueError("not json"))

        with (
            patch.object(reporter._http, "post", return_value=resp),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event()])

        assert "reporter.ingest_response_unparseable" in caplog.text
        assert "exc_type=ValueError" in caplog.text
        reporter._http.close()

    def test_rejected_longer_than_batch_fails_open(self, caplog: pytest.LogCaptureFixture) -> None:
        # A server rejecting more events than were submitted is contract-
        # violating: take the fail-open unparseable path, never emit
        # per-group warnings from a body that cannot be trusted.
        reporter = _quiet_sync_reporter()
        body = {"ingested": 0, "rejected": [_rejection(i) for i in range(3)]}

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event() for _ in range(2)])

        unparseable = [
            record.getMessage()
            for record in caplog.records
            if "reporter.ingest_response_unparseable" in record.getMessage()
        ]
        assert len(unparseable) == 1
        assert "exc_type=ValueError" in unparseable[0]
        assert "reporter.ingest_events_rejected" not in caplog.text
        reporter._http.close()

    def test_raw_control_bytes_in_echoed_values_are_escaped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A compliant server repr-escapes control characters before echoing
        # them back; if a regression sends RAW bytes, the SDK escapes them
        # itself so customer logs carry no ANSI sequences or forged lines.
        reporter = _quiet_sync_reporter()
        body = {
            "ingested": 0,
            "rejected": [
                _rejection(0, model="vendor\x1b[31m-x", message="tier 'spot\x1b[31m\n' unknown")
            ],
        }

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response(body)),
            caplog.at_level("WARNING"),
        ):
            reporter._send_batch([_make_event()])

        rejection_logs = [
            record.getMessage()
            for record in caplog.records
            if "reporter.ingest_events_rejected" in record.getMessage()
        ]
        assert len(rejection_logs) == 1
        assert "\\x1b" in rejection_logs[0]
        assert "\\n" in rejection_logs[0]
        assert "\x1b" not in rejection_logs[0]
        assert "\n" not in rejection_logs[0]
        reporter._http.close()

    def test_persistent_unparseable_bodies_escalate_to_error_then_reset(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Mirrors the confirm-failure escalation: the 10th consecutive
        # unparseable body logs at ERROR; one parseable body — even a clean
        # one — resets the counter.
        reporter = _quiet_sync_reporter()

        with (
            patch.object(reporter._http, "post", return_value=_accepted_response({"ingested": 1})),
            caplog.at_level("WARNING"),
        ):
            for _ in range(10):
                reporter._send_batch([_make_event()])

        persistent = [
            record
            for record in caplog.records
            if "reporter.ingest_response_unparseable_persistent" in record.getMessage()
        ]
        assert len(persistent) == 1
        assert persistent[0].levelname == "ERROR"
        assert "consecutive_failures=10" in persistent[0].getMessage()
        assert reporter._consecutive_unparseable_responses == 10

        with patch.object(
            reporter._http,
            "post",
            return_value=_accepted_response({"ingested": 1, "rejected": []}),
        ):
            reporter._send_batch([_make_event()])

        assert reporter._consecutive_unparseable_responses == 0
        reporter._http.close()

    def test_raising_logging_stack_does_not_mislabel_durable_batch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The batch IS durable server-side once the 202 lands — if the host's
        # logging stack raises during rejection emission (e.g. a
        # user-installed Filter), _send_batch must neither raise nor report
        # a send failure for a batch that was sent.
        reporter = _quiet_sync_reporter()
        body = {"ingested": 0, "rejected": [_rejection(0)]}

        class _RaisingFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if "ingest_events_rejected" in record.getMessage():
                    raise RuntimeError("user filter exploded")
                return True

        log_filter = _RaisingFilter()
        reporter_logger = logging.getLogger("solwyn.reporter")
        reporter_logger.addFilter(log_filter)
        try:
            with (
                patch.object(reporter._http, "post", return_value=_accepted_response(body)),
                caplog.at_level("WARNING"),
            ):
                # Must not raise despite the exploding filter.
                reporter._send_batch([_make_event()])
        finally:
            reporter_logger.removeFilter(log_filter)

        assert "Failed to send metadata batch" not in caplog.text
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
