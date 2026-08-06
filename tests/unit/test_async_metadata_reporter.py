"""Tests for AsyncMetadataReporter lifecycle — start, flush, close, context manager."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, _accepted_response, call_uuid

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName
from solwyn.reporter import AsyncMetadataReporter


def _make_event(**overrides) -> MetadataEvent:
    """Create a MetadataEvent with sensible test defaults."""
    defaults = {
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 200.0,
        "status": "success",
        "is_model_fallback": False,
        "sdk_instance_id": "test-instance-001",
        "timestamp": datetime.now(UTC),
        "call_id": call_uuid("call_async_reporter_event"),
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


def _make_confirm_request(**overrides) -> BudgetConfirmRequest:
    defaults = {
        "reservation_id": "res_123",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "call_id": call_uuid("call_async_confirm"),
        "token_details": TokenDetails(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return BudgetConfirmRequest(**defaults)


def _ok_response() -> MagicMock:
    """A 2xx httpx.Response stand-in: raise_for_status is a no-op.

    The response is sync (httpx.Response.raise_for_status() is sync even on the
    async client), so use MagicMock — not AsyncMock — per tests/CLAUDE.md.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    # Clean-batch body keeps lifecycle tests on the happy path — an
    # unconfigured .json mock would route them through the fail-open
    # unparseable branch and skew the consecutive-failure counter.
    resp.json = MagicMock(return_value={"ingested": 0, "rejected": []})
    return resp


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


# ---------------------------------------------------------------------------
# Lifecycle: start -> flush -> close
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterLifecycle:
    """Test the full start/report/flush/close lifecycle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_creates_flush_task(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()

        assert reporter._flush_task is not None
        assert reporter._shutdown_event is not None
        assert not reporter._shutdown_event.is_set()

        # Clean up
        reporter._shutdown_event.set()
        await reporter._flush_task
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_flushes_remaining_events(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            batch_size=10,
            flush_interval=60.0,
        )
        reporter.start()

        for _ in range(3):
            reporter.report(_make_event())

        assert len(reporter._queue) == 3

        mock_response = _ok_response()
        with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
            await reporter.close()

        # Should have flushed all events
        assert len(reporter._queue) == 0
        assert mock_post.call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_sets_shutdown_event(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()

        with patch.object(reporter._http, "post", new_callable=AsyncMock):
            await reporter.close()

        assert reporter._shutdown_event.is_set()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_flushes_remaining_confirms(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()
        reporter.report_confirm(_make_confirm_request())

        with patch.object(
            reporter._http, "post", new_callable=AsyncMock, return_value=_ok_response()
        ) as mock_post:
            await reporter.close()

        mock_post.assert_called_once()
        assert "budgets/confirm" in mock_post.call_args.args[0]
        assert mock_post.call_args.kwargs["json"]["call_id"] == call_uuid("call_async_confirm")


# ---------------------------------------------------------------------------
# Batch sending
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterSendBatch:
    """_send_batch posts events to the ingest endpoint."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_posts_to_ingest(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        batch = [_make_event() for _ in range(3)]
        mock_response = _ok_response()

        with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
            await reporter._send_batch(batch)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "metadata/ingest" in call_kwargs[0][0]
        assert len(call_kwargs[1]["json"]) == 3
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_omits_none_fields_from_payload(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with patch.object(
            reporter._http, "post", new_callable=AsyncMock, return_value=_ok_response()
        ) as mock_post:
            await reporter._send_batch([_make_event(service_tier=None)])

        payload = mock_post.call_args.kwargs["json"][0]
        assert "service_tier" not in payload
        assert "token_details" not in payload
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_includes_service_tier_when_present(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with patch.object(
            reporter._http, "post", new_callable=AsyncMock, return_value=_ok_response()
        ) as mock_post:
            await reporter._send_batch([_make_event(service_tier="priority")])

        payload = mock_post.call_args.kwargs["json"][0]
        assert payload["service_tier"] == "priority"
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_includes_agent_run_fields_when_set(self) -> None:
        # Async reporter must also propagate agent_run_* into the wire payload.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        event = _make_event(
            agent_run_id="run_test_xyz",
            parent_agent_run_id="run_parent_xyz",
            agent_run_name="async-batch",
        )
        with patch.object(
            reporter._http, "post", new_callable=AsyncMock, return_value=_ok_response()
        ) as mock_post:
            await reporter._send_batch([event])

        payload = mock_post.call_args.kwargs["json"][0]
        assert payload["agent_run_id"] == "run_test_xyz"
        assert payload["parent_agent_run_id"] == "run_parent_xyz"
        assert payload["agent_run_name"] == "async-batch"
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_omits_agent_run_fields_when_none(self) -> None:
        # When unset, async reporter must omit the keys so server-side
        # per-day fallback id engages.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with patch.object(
            reporter._http, "post", new_callable=AsyncMock, return_value=_ok_response()
        ) as mock_post:
            await reporter._send_batch([_make_event()])

        payload = mock_post.call_args.kwargs["json"][0]
        assert "agent_run_id" not in payload
        assert "parent_agent_run_id" not in payload
        assert "agent_run_name" not in payload
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_swallows_errors(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        batch = [_make_event()]

        with patch.object(reporter._http, "post", side_effect=RuntimeError("fail")):
            # Should not raise
            await reporter._send_batch(batch)

        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_4xx_logged_type_only_and_survives(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A 4xx on the ingest POST must surface via raise_for_status, be caught,
        # logged by class name only (no body), and not propagate.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with (
            patch.object(
                reporter._http, "post", new_callable=AsyncMock, return_value=_error_response(422)
            ),
            caplog.at_level("WARNING"),
        ):
            # Should not raise despite the 422.
            await reporter._send_batch([_make_event()])

        assert "HTTPStatusError" in caplog.text
        # Privacy: never log the response body — only the exception class name.
        assert "422 Unprocessable" not in caplog.text
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_logs_aggregated_rejections(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Async path shares the sans-I/O rejection parsing: one aggregated
        # WARNING per distinct (code, model), rejected events stay dropped.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        body = {
            "ingested": 1,
            "rejected": [
                {"index": 0, "code": "unknown_model", "model": "vendor-x-1", "message": "m"},
                {"index": 1, "code": "unknown_model", "model": "vendor-x-1", "message": "m"},
                {"index": 2, "code": "unknown_service_tier", "model": "gpt-5.5", "message": "m2"},
            ],
        }

        with (
            patch.object(
                reporter._http,
                "post",
                new_callable=AsyncMock,
                return_value=_accepted_response(body),
            ),
            caplog.at_level("WARNING"),
        ):
            await reporter._send_batch([_make_event() for _ in range(3)])

        rejection_logs = [
            record.getMessage()
            for record in caplog.records
            if "reporter.ingest_events_rejected" in record.getMessage()
        ]
        assert len(rejection_logs) == 2
        assert any("code=unknown_model model=vendor-x-1 count=2" in line for line in rejection_logs)
        assert any(
            "code=unknown_service_tier model=gpt-5.5 count=1" in line for line in rejection_logs
        )
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rejected_events_are_not_requeued(self) -> None:
        # Rejected events are terminal for the flush: the queue stays drained
        # and a second flush has nothing to send.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        reporter.report(_make_event())
        reporter.report(_make_event())
        body = {
            "ingested": 1,
            "rejected": [
                {"index": 0, "code": "unknown_model", "model": "vendor-x-1", "message": "m"},
            ],
        }

        with patch.object(
            reporter._http,
            "post",
            new_callable=AsyncMock,
            return_value=_accepted_response(body),
        ):
            await reporter._flush_remaining()

        assert len(reporter._queue) == 0

        with patch.object(reporter._http, "post", new_callable=AsyncMock) as mock_post:
            await reporter._flush_remaining()
        assert mock_post.call_count == 0
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_empty_rejected_list_logs_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The server always sends rejected (possibly []) — the clean-batch
        # path must stay silent, exactly like pre-v0.1.7.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with (
            patch.object(
                reporter._http,
                "post",
                new_callable=AsyncMock,
                return_value=_accepted_response({"ingested": 3, "rejected": []}),
            ),
            caplog.at_level("WARNING"),
        ):
            await reporter._send_batch([_make_event() for _ in range(3)])

        assert not caplog.records
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_invalid_json_202_body_fails_open(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A body that is not JSON at all (resp.json() raises) takes the same
        # fail-open path.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 202
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(side_effect=ValueError("not json"))

        with (
            patch.object(reporter._http, "post", new_callable=AsyncMock, return_value=resp),
            caplog.at_level("WARNING"),
        ):
            await reporter._send_batch([_make_event()])

        assert "reporter.ingest_response_unparseable" in caplog.text
        assert "exc_type=ValueError" in caplog.text
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_non_dict_rejection_entry_fails_open(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A rejected entry that is not a dict is contract-violating: fall back
        # to count-only acknowledgment, never raise into the caller's loop.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with (
            patch.object(
                reporter._http,
                "post",
                new_callable=AsyncMock,
                return_value=_accepted_response({"ingested": 1, "rejected": ["not-a-dict"]}),
            ),
            caplog.at_level("WARNING"),
        ):
            await reporter._send_batch([_make_event()])

        assert "reporter.ingest_response_unparseable" in caplog.text
        assert "exc_type=TypeError" in caplog.text
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_send_batch_malformed_202_body_fails_open(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Fail-open: a contract-violating body must never raise into the
        # caller's event loop — fall back to count-only acknowledgment.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with (
            patch.object(
                reporter._http,
                "post",
                new_callable=AsyncMock,
                return_value=_accepted_response({"ingested": 1}),
            ),
            caplog.at_level("WARNING"),
        ):
            await reporter._send_batch([_make_event()])

        assert "reporter.ingest_response_unparseable" in caplog.text
        assert "exc_type=KeyError" in caplog.text
        await reporter._http.aclose()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterContextManager:
    """async with AsyncMetadataReporter starts and closes correctly."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager_starts_and_closes(self) -> None:
        with patch.object(AsyncMetadataReporter, "_send_batch", new_callable=AsyncMock):
            async with AsyncMetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
                flush_interval=60.0,
            ) as reporter:
                reporter.report(_make_event())
                assert reporter._flush_task is not None

        # After exit, shutdown should be set
        assert reporter._shutdown_event.is_set()


# ---------------------------------------------------------------------------
# Batch size flush
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterBatchFlush:
    """Events are flushed in correct batch sizes."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_remaining_batches_correctly(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            batch_size=3,
        )

        for _ in range(5):
            reporter.report(_make_event())

        mock_response = _ok_response()
        with patch.object(reporter._http, "post", return_value=mock_response) as mock_post:
            await reporter._flush_remaining()

        # 5 events / batch_size 3 = 2 batches (3 + 2)
        assert mock_post.call_count == 2
        assert len(reporter._queue) == 0
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_remaining_sends_queued_confirms_before_success_events(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        call_id = call_uuid("call_async_stream_settlement")
        reporter.report(_make_event(call_id=call_id))
        reporter.report_confirm(_make_confirm_request(call_id=call_id))

        with patch.object(
            reporter._http, "post", new_callable=AsyncMock, return_value=_ok_response()
        ) as mock_post:
            await reporter._flush_remaining()

        assert [call.args[0] for call in mock_post.call_args_list] == [
            "https://api.test.solwyn.ai/api/v1/budgets/confirm",
            "https://api.test.solwyn.ai/api/v1/metadata/ingest",
        ]
        assert mock_post.call_args_list[0].kwargs["json"]["call_id"] == call_id
        assert mock_post.call_args_list[1].kwargs["json"][0]["call_id"] == call_id
        assert len(reporter._queue) == 0
        assert len(reporter._confirm_queue) == 0
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_settlement_enqueued_during_metadata_send_waits_for_confirm_first(
        self,
    ) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            batch_size=1,
        )
        older_call_id = call_uuid("call_async_older_metadata")
        settlement_call_id = call_uuid("call_async_mid_flush_settlement")
        reporter.report(_make_event(call_id=older_call_id))
        settlement_confirm = _make_confirm_request(call_id=settlement_call_id)
        settlement_event = _make_event(call_id=settlement_call_id)
        calls: list[tuple[str, Any]] = []
        did_enqueue = False

        async def post(url: str, **kwargs: Any) -> MagicMock:
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
            return _ok_response()

        with patch.object(reporter._http, "post", new=post):
            await reporter._flush_remaining()
            await reporter._flush_remaining()

        settlement_posts = []
        for url, payload in calls:
            if "budgets/confirm" in url and payload["call_id"] == settlement_call_id:
                settlement_posts.append("confirm")
            elif "metadata/ingest" in url and payload[0]["call_id"] == settlement_call_id:
                settlement_posts.append("metadata")

        assert settlement_posts == ["confirm", "metadata"]
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_remaining_confirm_4xx_logged_type_only_and_survives(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A 4xx on the confirm POST must surface via raise_for_status, be caught,
        # logged by class name only, drain from the queue, and not propagate.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        reporter.report_confirm(_make_confirm_request())

        with (
            patch.object(
                reporter._http, "post", new_callable=AsyncMock, return_value=_error_response(422)
            ),
            caplog.at_level("WARNING"),
        ):
            # Should not raise despite the 422.
            await reporter._flush_remaining()

        assert "reporter.confirm_send_failed" in caplog.text
        assert "HTTPStatusError" in caplog.text
        # The flush loop survives and drains the confirm queue.
        assert len(reporter._confirm_queue) == 0
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_flush_remaining_confirm_persistent_failures_escalate_to_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A retrying head PARKS its queue (FIFO), so a persistent outage
        # accrues one consecutive failure per flush cycle, not per queued item.
        clock = {"t": 1000.0}
        monkeypatch.setattr("solwyn.reporter._monotonic", lambda: clock["t"])
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            max_send_attempts=11,
        )
        reporter.report_confirm(_make_confirm_request())

        with (
            patch.object(
                reporter._http, "post", new_callable=AsyncMock, return_value=_error_response(503)
            ),
            caplog.at_level("ERROR"),
        ):
            for _ in range(10):
                await reporter._flush_remaining()
                clock["t"] += 61.0  # past the backoff cap so the head is due

        assert reporter._consecutive_confirm_failures == 10
        assert "reporter.confirm_send_persistent_failure" in caplog.text
        assert "consecutive_failures=10" in caplog.text
        assert "HTTPStatusError" in caplog.text
        await reporter._http.aclose()


# ---------------------------------------------------------------------------
# Auto-start on first enqueue (sync/async parity)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterAutoStart:
    """First enqueue auto-starts the flush loop; start() is idempotent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_report_auto_starts_flush_task(self) -> None:
        # A reporter constructed without ``async with`` must not queue events
        # silently: the first report() inside a running loop starts the flush
        # loop so the batch actually gets delivered.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        assert reporter._flush_task is None

        reporter.report(_make_event())

        assert reporter._flush_task is not None
        assert not reporter._flush_task.done()
        assert reporter._shutdown_event is not None
        assert len(reporter._queue) == 1

        # Clean up
        reporter._shutdown_event.set()
        await reporter._flush_task
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_report_confirm_auto_starts_flush_task(self) -> None:
        # Budget confirms are settlement data — they must not queue silently
        # either, so report_confirm() auto-starts the flush loop as well.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        assert reporter._flush_task is None

        reporter.report_confirm(_make_confirm_request())

        assert reporter._flush_task is not None
        assert not reporter._flush_task.done()
        assert len(reporter._confirm_queue) == 1

        # Clean up
        reporter._shutdown_event.set()
        await reporter._flush_task
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_report_settlement_auto_starts_flush_task(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        assert reporter._flush_task is None

        reporter.report_settlement(_make_confirm_request(), _make_event())

        assert reporter._flush_task is not None
        assert not reporter._flush_task.done()
        assert len(reporter._settlement_queue) == 1

        # Clean up
        reporter._shutdown_event.set()
        await reporter._flush_task
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_is_idempotent_returns_same_task(self) -> None:
        # A second start() must not orphan the live flush task or replace the
        # shutdown event.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()
        first_task = reporter._flush_task
        first_shutdown = reporter._shutdown_event

        reporter.start()

        assert reporter._flush_task is first_task
        assert reporter._shutdown_event is first_shutdown

        # Clean up
        assert reporter._shutdown_event is not None
        reporter._shutdown_event.set()
        assert reporter._flush_task is not None
        await reporter._flush_task
        await reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_after_close_raises(self) -> None:
        # Restarting a closed reporter is a programming error — fail loud
        # (project rule: RuntimeError, never assert).
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()
        with patch.object(reporter._http, "post", new_callable=AsyncMock):
            await reporter.close()

        with pytest.raises(RuntimeError):
            reporter.start()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enqueue_after_close_is_dropped(self) -> None:
        # After close(), enqueues are silently dropped (matches the sync
        # reporter): nothing queues, no new flush task, no exception.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )
        reporter.start()
        with patch.object(reporter._http, "post", new_callable=AsyncMock):
            await reporter.close()
        task_after_close = reporter._flush_task

        reporter.report(_make_event())
        reporter.report_confirm(_make_confirm_request())
        reporter.report_settlement(_make_confirm_request(), _make_event())

        assert len(reporter._queue) == 0
        assert len(reporter._confirm_queue) == 0
        assert len(reporter._settlement_queue) == 0
        # No new flush task was spawned by the dropped enqueues.
        assert reporter._flush_task is task_after_close

    @pytest.mark.unit
    def test_report_without_event_loop_queues_and_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Plain sync context (no running loop): report() must not raise, the
        # event stays queued, no flush task is created, and the "no event loop"
        # warning fires exactly once per instance across repeated calls.
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=60.0,
        )

        with caplog.at_level("WARNING"):
            reporter.report(_make_event())
            reporter.report(_make_event())

        assert len(reporter._queue) == 2
        assert reporter._flush_task is None
        no_loop_warnings = [
            record
            for record in caplog.records
            if "reporter.enqueue_without_event_loop" in record.getMessage()
        ]
        assert len(no_loop_warnings) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_flushes_auto_started_events(self) -> None:
        # Events queued via auto-start (no explicit start()/async with) are
        # delivered on close().
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            batch_size=10,
            flush_interval=60.0,
        )

        with patch.object(reporter, "_send_batch", new_callable=AsyncMock) as mock_send:
            for _ in range(3):
                reporter.report(_make_event())
            assert reporter._flush_task is not None  # auto-started
            await reporter.close()

        mock_send.assert_called_once()
        assert len(mock_send.call_args.args[0]) == 3
        assert len(reporter._queue) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_with_and_report_compose_to_single_task(self) -> None:
        # __aenter__ starts the flush task; a subsequent report() must reuse it
        # (auto-start is idempotent), not spawn a second.
        with patch.object(AsyncMetadataReporter, "_send_batch", new_callable=AsyncMock):
            async with AsyncMetadataReporter(
                "https://api.test.solwyn.ai",
                VALID_API_KEY,
                flush_interval=60.0,
            ) as reporter:
                task_after_enter = reporter._flush_task
                assert task_after_enter is not None

                reporter.report(_make_event())

                assert reporter._flush_task is task_after_enter

        assert reporter._shutdown_event is not None
        assert reporter._shutdown_event.is_set()
