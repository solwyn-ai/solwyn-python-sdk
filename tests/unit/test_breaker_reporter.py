"""Focused tests for periodic circuit-breaker snapshot reporting."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID, _accepted_response, call_uuid

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, CallStatus, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.client import Solwyn
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter

SDK_INSTANCE_ID = "sdk-instance-breaker-reporter"
BREAKER_FIELDS = {
    "provider",
    "state",
    "failure_count",
    "success_count",
    "reported_at",
    "sdk_instance_id",
}


def _ok_response() -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 202
    response.raise_for_status = MagicMock()
    return response


def _error_response() -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 503
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "unavailable",
            request=MagicMock(spec=httpx.Request),
            response=response,
        )
    )
    return response


def _event() -> MetadataEvent:
    return MetadataEvent(
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        input_tokens=10,
        output_tokens=5,
        latency_ms=12.0,
        status=CallStatus.SUCCESS,
        is_model_fallback=False,
        sdk_instance_id=SDK_INSTANCE_ID,
        timestamp=datetime.now(UTC),
        call_id=call_uuid("call-breaker-report"),
    )


def _confirm() -> BudgetConfirmRequest:
    return BudgetConfirmRequest(
        reservation_id="res-breaker-report",
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        token_details=TokenDetails(input_tokens=10, output_tokens=5),
        call_id=call_uuid("call-breaker-report"),
    )


def _quiet_sync_reporter(**kwargs: object) -> MetadataReporter:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            **kwargs,
        )
    reporter._thread.join(timeout=2.0)
    return reporter


@pytest.fixture
def reporter_with_two_providers() -> Iterator[
    tuple[MetadataReporter, dict[ProviderName, CircuitBreaker], MagicMock]
]:
    breakers = {
        ProviderName.OPENAI: CircuitBreaker(failure_threshold=1),
        ProviderName.ANTHROPIC: CircuitBreaker(failure_threshold=1),
    }
    reporter = _quiet_sync_reporter(
        breaker_snapshots=lambda: [
            (provider, breaker.get_state()) for provider, breaker in breakers.items()
        ],
        sdk_instance_id=SDK_INSTANCE_ID,
    )
    reporter.observe_project_id(VALID_PROJECT_ID)
    with patch.object(reporter._http, "post", return_value=_ok_response()) as post:
        yield reporter, breakers, post
    reporter._http.close()


def _async_reporter_with_two_providers() -> tuple[
    AsyncMetadataReporter, dict[ProviderName, CircuitBreaker]
]:
    breakers = {
        ProviderName.OPENAI: CircuitBreaker(failure_threshold=1),
        ProviderName.ANTHROPIC: CircuitBreaker(failure_threshold=1),
    }
    reporter = AsyncMetadataReporter(
        "https://api.test.solwyn.ai",
        VALID_API_KEY,
        breaker_snapshots=lambda: [
            (provider, breaker.get_state()) for provider, breaker in breakers.items()
        ],
        sdk_instance_id=SDK_INSTANCE_ID,
    )
    reporter.observe_project_id(VALID_PROJECT_ID)
    return reporter, breakers


@pytest.mark.unit
class TestSyncBreakerReporter:
    def test_steady_state_sends_nothing_after_first_cycle(
        self,
        reporter_with_two_providers: tuple[
            MetadataReporter, dict[ProviderName, CircuitBreaker], MagicMock
        ],
    ) -> None:
        reporter, _breakers, post = reporter_with_two_providers

        reporter._flush_breaker_reports()
        first = post.call_count
        assert first == 2
        reporter._flush_breaker_reports()

        assert post.call_count == first

    def test_state_change_sends_only_changed_provider(
        self,
        reporter_with_two_providers: tuple[
            MetadataReporter, dict[ProviderName, CircuitBreaker], MagicMock
        ],
    ) -> None:
        reporter, breakers, post = reporter_with_two_providers
        reporter._flush_breaker_reports()
        before = post.call_count

        breakers[ProviderName.OPENAI].record_failure()
        reporter._flush_breaker_reports()

        assert post.call_count == before + 1

    def test_heartbeat_resends_all(
        self,
        reporter_with_two_providers: tuple[
            MetadataReporter, dict[ProviderName, CircuitBreaker], MagicMock
        ],
    ) -> None:
        reporter, _breakers, post = reporter_with_two_providers
        reporter._flush_breaker_reports()
        before = post.call_count

        # -inf, not 0.0: monotonic's epoch is arbitrary (boot time on Linux), so
        # 0.0 only makes the heartbeat due once the machine is heartbeat-old.
        reporter._breaker_heartbeat_at = float("-inf")
        reporter._flush_breaker_reports()

        assert post.call_count == before + 2

    def test_failed_send_retries_next_cycle(
        self,
        reporter_with_two_providers: tuple[
            MetadataReporter, dict[ProviderName, CircuitBreaker], MagicMock
        ],
    ) -> None:
        reporter, _breakers, post = reporter_with_two_providers
        post.side_effect = ConnectionError("down")
        reporter._flush_breaker_reports()

        post.side_effect = None
        reporter._flush_breaker_reports()

        assert post.call_count == 4

    def test_force_sends_regardless_of_gating(
        self,
        reporter_with_two_providers: tuple[
            MetadataReporter, dict[ProviderName, CircuitBreaker], MagicMock
        ],
    ) -> None:
        reporter, _breakers, post = reporter_with_two_providers
        reporter._flush_breaker_reports()
        before = post.call_count

        reporter._flush_breaker_reports(force=True)

        assert post.call_count == before + 2

    def test_reports_due_false_in_steady_state(
        self,
        reporter_with_two_providers: tuple[
            MetadataReporter, dict[ProviderName, CircuitBreaker], MagicMock
        ],
    ) -> None:
        reporter, _breakers, _post = reporter_with_two_providers
        reporter._flush_breaker_reports()

        assert reporter._breaker_reports_due() is False

    def test_blocked_breaker_cycle_does_not_block_later_metadata_or_spawn_more_cycles(
        self,
    ) -> None:
        breaker_started = threading.Event()
        release_breaker = threading.Event()
        first_metadata_sent = threading.Event()
        second_metadata_sent = threading.Event()
        breaker_calls = 0
        metadata_calls = 0
        reporter = MetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=0.01,
            breaker_snapshots=lambda: [(ProviderName.OPENAI, CircuitBreaker().get_state())],
            sdk_instance_id=SDK_INSTANCE_ID,
        )

        def send(url: str, **_kwargs: object) -> MagicMock:
            nonlocal breaker_calls, metadata_calls
            if url.endswith("/breaker-reports"):
                breaker_calls += 1
                breaker_started.set()
                assert release_breaker.wait(timeout=2.0)
                return _ok_response()
            if url.endswith("/metadata/ingest"):
                metadata_calls += 1
                if metadata_calls == 1:
                    first_metadata_sent.set()
                else:
                    second_metadata_sent.set()
                return _accepted_response({"ingested": 1, "rejected": []})
            return _ok_response()

        with patch.object(reporter._http, "post", side_effect=send):
            try:
                reporter.observe_project_id(VALID_PROJECT_ID)
                assert breaker_started.wait(timeout=1.0)
                reporter.report(_event())

                assert first_metadata_sent.wait(timeout=1.0)
                reporter.report(_event())
                assert second_metadata_sent.wait(timeout=1.0)
                assert breaker_calls == 1
            finally:
                release_breaker.set()
                reporter.close()

    def test_close_waits_for_active_breaker_cycle_then_forces_full_final_cycle(
        self,
    ) -> None:
        breaker_started = threading.Event()
        release_breaker = threading.Event()
        close_finished = threading.Event()
        sent_providers: list[str] = []
        openai = CircuitBreaker(failure_threshold=1)
        anthropic = CircuitBreaker(failure_threshold=1)
        reporter = _quiet_sync_reporter(
            breaker_snapshots=lambda: [
                (ProviderName.OPENAI, openai.get_state()),
                (ProviderName.ANTHROPIC, anthropic.get_state()),
            ],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        real_close = reporter._http.close

        def send(url: str, **kwargs: object) -> MagicMock:
            if url.endswith("/breaker-reports"):
                payload = kwargs["json"]
                assert isinstance(payload, dict)
                sent_providers.append(str(payload["provider"]))
                if len(sent_providers) == 1:
                    breaker_started.set()
                    assert release_breaker.wait(timeout=2.0)
            return _ok_response()

        def close_reporter() -> None:
            reporter.close()
            close_finished.set()

        close_thread: threading.Thread | None = None
        try:
            with (
                patch.object(reporter._http, "post", return_value=_ok_response()) as post,
                patch.object(reporter._http, "close") as http_close,
            ):
                reporter.observe_project_id(VALID_PROJECT_ID)
                reporter._flush_breaker_reports()
                post.reset_mock()
                openai.record_failure()
                post.side_effect = send
                reporter._start_breaker_cycle()
                assert breaker_started.wait(timeout=1.0)
                close_thread = threading.Thread(target=close_reporter)
                close_thread.start()

                assert not close_finished.wait(timeout=0.05)
                assert sent_providers == ["openai"]
                http_close.assert_not_called()
                release_breaker.set()
                assert close_finished.wait(timeout=2.0)
                close_thread.join(timeout=2.0)

                assert sent_providers == ["openai", "openai", "anthropic"]
                http_close.assert_called_once_with()
        finally:
            release_breaker.set()
            if close_thread is not None:
                close_thread.join(timeout=2.0)
            real_close()

    def test_close_deadline_closes_http_with_active_breaker_worker_stuck(self) -> None:
        breaker_started = threading.Event()
        release_breaker = threading.Event()
        reporter = _quiet_sync_reporter(
            breaker_snapshots=lambda: [(ProviderName.OPENAI, CircuitBreaker().get_state())],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)
        real_close = reporter._http.close

        def send(url: str, **_kwargs: object) -> MagicMock:
            if url.endswith("/breaker-reports"):
                breaker_started.set()
                assert release_breaker.wait(timeout=2.0)
            return _ok_response()

        worker: threading.Thread | None = None
        try:
            with (
                patch.object(reporter._http, "post", side_effect=send),
                patch.object(reporter._http, "close") as http_close,
            ):
                worker = reporter._start_breaker_cycle()
                assert worker is not None
                assert breaker_started.wait(timeout=1.0)

                started_at = time.monotonic()
                reporter.close(timeout=0.01)
                elapsed = time.monotonic() - started_at

                assert elapsed < 0.5
                assert worker.is_alive()
                http_close.assert_called_once_with()
        finally:
            release_breaker.set()
            if worker is not None:
                worker.join(timeout=2.0)
            real_close()

    def test_client_supplies_distinct_provider_snapshots_and_instance_config(self) -> None:
        openai = MagicMock()
        openai.__class__.__module__ = "openai._client"
        openai.__class__.__name__ = "OpenAI"
        anthropic = MagicMock()
        anthropic.__class__.__module__ = "anthropic._client"
        anthropic.__class__.__name__ = "Anthropic"

        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            solwyn = Solwyn(
                openai,
                api_key=VALID_API_KEY,
                model="gpt-5.5",
                fallback=[(anthropic, "claude-sonnet-5")],
                breaker_reporting_enabled=False,
            )
        solwyn._solwyn_reporter._thread.join(timeout=2.0)

        assert solwyn._solwyn_reporter._breaker_snapshots is not None
        snapshots = solwyn._solwyn_reporter._breaker_snapshots()
        assert {provider for provider, _snapshot in snapshots} == {
            ProviderName.OPENAI,
            ProviderName.ANTHROPIC,
        }
        assert all(snapshot.model_config["frozen"] for _provider, snapshot in snapshots)
        assert solwyn._solwyn_reporter._sdk_instance_id == solwyn._solwyn_sdk_instance_id
        assert solwyn._solwyn_reporter._breaker_reporting_enabled is False

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_next_flush_uses_current_state_and_wall_clock_only(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)
        reporter = _quiet_sync_reporter(
            breaker_snapshots=lambda: [(ProviderName.OPENAI, breaker.get_state())],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)
        before = datetime.now(UTC)

        with patch.object(reporter._http, "post", return_value=_ok_response()) as post:
            reporter._flush_breaker_reports()
            breaker.record_failure()
            reporter._flush_breaker_reports()

        after = datetime.now(UTC)
        payloads = [call.kwargs["json"] for call in post.call_args_list]
        assert [payload["state"] for payload in payloads] == ["closed", "open"]
        assert [payload["failure_count"] for payload in payloads] == [0, 1]
        assert all(set(payload) == BREAKER_FIELDS for payload in payloads)
        assert all(payload["sdk_instance_id"] == SDK_INSTANCE_ID for payload in payloads)
        for payload in payloads:
            reported_at = datetime.fromisoformat(payload["reported_at"].replace("Z", "+00:00"))
            assert before <= reported_at <= after
            assert reported_at.tzinfo is not None

        reporter._http.close()

    def test_cycle_timestamp_is_fixed_before_first_provider_io(self) -> None:
        first_provider_started = threading.Event()
        release_first_provider = threading.Event()
        captured_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        late_at = datetime(2026, 7, 14, 12, 0, 5, tzinfo=UTC)
        payloads: list[dict[str, object]] = []
        reporter = _quiet_sync_reporter(
            breaker_snapshots=lambda: [
                (ProviderName.OPENAI, CircuitBreaker().get_state()),
                (ProviderName.ANTHROPIC, CircuitBreaker().get_state()),
            ],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)

        def send(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]
            assert isinstance(payload, dict)
            payloads.append(payload)
            if payload["provider"] == "openai":
                first_provider_started.set()
                assert release_first_provider.wait(timeout=2.0)
            return _ok_response()

        worker = threading.Thread(target=reporter._flush_breaker_reports)
        try:
            with (
                patch("solwyn.reporter.datetime") as clock,
                patch.object(reporter._http, "post", side_effect=send),
            ):
                clock.now.side_effect = [captured_at, late_at]
                worker.start()
                assert first_provider_started.wait(timeout=1.0)
                assert clock.now.call_count == 1
                release_first_provider.set()
                worker.join(timeout=2.0)

                assert not worker.is_alive()
                assert len(payloads) == 2
                assert {payload["reported_at"] for payload in payloads} == {
                    captured_at.isoformat().replace("+00:00", "Z")
                }
                clock.now.assert_called_once_with(UTC)
        finally:
            release_first_provider.set()
            worker.join(timeout=2.0)
            reporter._http.close()

    def test_failure_isolated_from_confirms_metadata_and_later_providers(self) -> None:
        openai = CircuitBreaker()
        anthropic = CircuitBreaker()
        reporter = _quiet_sync_reporter(
            breaker_snapshots=lambda: [
                (ProviderName.OPENAI, openai.get_state()),
                (ProviderName.ANTHROPIC, anthropic.get_state()),
            ],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)
        reporter.report_confirm(_confirm())
        reporter.report(_event())

        def send(url: str, **kwargs: object) -> MagicMock:
            if url.endswith("/budgets/confirm"):
                return _ok_response()
            if url.endswith("/metadata/ingest"):
                return _accepted_response({"ingested": 1, "rejected": []})
            if kwargs["json"]["provider"] == "openai":
                return _error_response()
            return _ok_response()

        with patch.object(reporter._http, "post", side_effect=send) as post:
            reporter._flush_remaining()
            reporter._flush_breaker_reports()

        urls = [call.args[0] for call in post.call_args_list]
        assert any(url.endswith("/budgets/confirm") for url in urls)
        assert any(url.endswith("/metadata/ingest") for url in urls)
        breaker_calls = [
            call for call in post.call_args_list if call.args[0].endswith("/breaker-reports")
        ]
        assert [call.kwargs["json"]["provider"] for call in breaker_calls] == [
            "openai",
            "anthropic",
        ]
        assert len(reporter._confirm_queue) == 0
        assert len(reporter._queue) == 0

        reporter._http.close()

    def test_unknown_project_does_not_collect_or_post(self) -> None:
        snapshots = MagicMock(side_effect=AssertionError("snapshot collection must be skipped"))
        reporter = _quiet_sync_reporter(
            breaker_snapshots=snapshots,
            sdk_instance_id=SDK_INSTANCE_ID,
        )

        with patch.object(reporter._http, "post") as post:
            reporter._flush_breaker_reports()

        snapshots.assert_not_called()
        post.assert_not_called()
        reporter._http.close()

    def test_disabled_reporting_does_not_collect_or_post(self) -> None:
        snapshots = MagicMock(side_effect=AssertionError("snapshot collection must be skipped"))
        reporter = _quiet_sync_reporter(
            breaker_snapshots=snapshots,
            sdk_instance_id=SDK_INSTANCE_ID,
            breaker_reporting_enabled=False,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)

        with patch.object(reporter._http, "post") as post:
            reporter._flush_breaker_reports()

        snapshots.assert_not_called()
        post.assert_not_called()
        reporter._http.close()

    def test_shutdown_signal_leaves_the_single_final_flush_to_close(self) -> None:
        reporter = _quiet_sync_reporter()
        shutdown = MagicMock()
        shutdown.is_set.side_effect = [False, True]
        shutdown.wait.return_value = True
        reporter._shutdown = shutdown

        with patch.object(reporter, "_flush_remaining") as flush:
            reporter._flush_loop()

        flush.assert_not_called()
        reporter._http.close()

    def test_idle_flush_tick_does_not_start_breaker_worker(self) -> None:
        reporter = _quiet_sync_reporter()
        shutdown = MagicMock()
        shutdown.is_set.side_effect = [False, True]
        shutdown.wait.return_value = False
        reporter._shutdown = shutdown

        with (
            patch.object(reporter, "_flush_remaining"),
            patch.object(reporter, "_breaker_reports_due", return_value=False) as due,
            patch.object(reporter, "_start_breaker_cycle") as start_cycle,
        ):
            reporter._flush_loop()

        due.assert_called_once_with()
        start_cycle.assert_not_called()
        reporter._http.close()

    def test_close_forces_final_breaker_cycle(self) -> None:
        reporter = _quiet_sync_reporter()

        with patch.object(reporter, "_start_breaker_cycle", return_value=None) as start_cycle:
            reporter.close()

        start_cycle.assert_called_once_with(
            during_shutdown=True,
            deadline=start_cycle.call_args.kwargs["deadline"],
            force=True,
        )


@pytest.mark.unit
class TestAsyncBreakerReporter:
    @pytest.mark.asyncio
    async def test_steady_state_sends_nothing_after_first_cycle(self) -> None:
        reporter, _breakers = _async_reporter_with_two_providers()

        with patch.object(
            reporter._http,
            "post",
            new_callable=AsyncMock,
            return_value=_ok_response(),
        ) as post:
            await reporter._flush_breaker_reports()
            first = post.call_count
            assert first == 2
            await reporter._flush_breaker_reports()

            assert post.call_count == first

        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_heartbeat_resends_all(self) -> None:
        reporter, _breakers = _async_reporter_with_two_providers()

        with patch.object(
            reporter._http,
            "post",
            new_callable=AsyncMock,
            return_value=_ok_response(),
        ) as post:
            await reporter._flush_breaker_reports()
            before = post.call_count
            # -inf, not 0.0: monotonic's epoch is arbitrary (boot time on
            # Linux), so 0.0 only makes the heartbeat due once the machine is
            # heartbeat-old.
            reporter._breaker_heartbeat_at = float("-inf")
            await reporter._flush_breaker_reports()

            assert post.call_count == before + 2

        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_failed_send_retries_next_cycle(self) -> None:
        reporter, _breakers = _async_reporter_with_two_providers()

        with patch.object(
            reporter._http,
            "post",
            new_callable=AsyncMock,
            return_value=_ok_response(),
        ) as post:
            post.side_effect = ConnectionError("down")
            await reporter._flush_breaker_reports()
            post.side_effect = None
            await reporter._flush_breaker_reports()

            assert post.call_count == 4

        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_force_sends_regardless_of_gating(self) -> None:
        reporter, _breakers = _async_reporter_with_two_providers()

        with patch.object(
            reporter._http,
            "post",
            new_callable=AsyncMock,
            return_value=_ok_response(),
        ) as post:
            await reporter._flush_breaker_reports()
            before = post.call_count
            await reporter._flush_breaker_reports(force=True)

            assert post.call_count == before + 2

        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_reports_due_false_in_steady_state(self) -> None:
        reporter, _breakers = _async_reporter_with_two_providers()

        with patch.object(
            reporter._http,
            "post",
            new_callable=AsyncMock,
            return_value=_ok_response(),
        ):
            await reporter._flush_breaker_reports()

            assert reporter._breaker_reports_due() is False

        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_blocked_breaker_cycle_does_not_block_later_metadata_or_spawn_more_cycles(
        self,
    ) -> None:
        breaker_started = asyncio.Event()
        release_breaker = asyncio.Event()
        first_metadata_sent = asyncio.Event()
        second_metadata_sent = asyncio.Event()
        breaker_calls = 0
        metadata_calls = 0
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            flush_interval=0.01,
            breaker_snapshots=lambda: [(ProviderName.OPENAI, CircuitBreaker().get_state())],
            sdk_instance_id=SDK_INSTANCE_ID,
        )

        async def send(url: str, **_kwargs: object) -> MagicMock:
            nonlocal breaker_calls, metadata_calls
            if url.endswith("/breaker-reports"):
                breaker_calls += 1
                breaker_started.set()
                await release_breaker.wait()
                return _ok_response()
            if url.endswith("/metadata/ingest"):
                metadata_calls += 1
                if metadata_calls == 1:
                    first_metadata_sent.set()
                else:
                    second_metadata_sent.set()
                return _accepted_response({"ingested": 1, "rejected": []})
            return _ok_response()

        with patch.object(reporter._http, "post", new=AsyncMock(side_effect=send)):
            try:
                reporter.observe_project_id(VALID_PROJECT_ID)
                reporter.start()
                await asyncio.wait_for(breaker_started.wait(), timeout=1.0)
                reporter.report(_event())
                try:
                    await asyncio.wait_for(first_metadata_sent.wait(), timeout=1.0)
                    reporter.report(_event())
                    await asyncio.wait_for(second_metadata_sent.wait(), timeout=1.0)
                    sent_while_blocked = True
                except TimeoutError:
                    sent_while_blocked = False

                assert sent_while_blocked
                assert breaker_calls == 1
            finally:
                release_breaker.set()
                await reporter.close()

    @pytest.mark.asyncio
    async def test_close_waits_for_active_breaker_cycle_then_forces_full_final_cycle(
        self,
    ) -> None:
        breaker_started = asyncio.Event()
        release_breaker = asyncio.Event()
        sent_providers: list[str] = []
        openai = CircuitBreaker(failure_threshold=1)
        anthropic = CircuitBreaker(failure_threshold=1)
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            breaker_snapshots=lambda: [
                (ProviderName.OPENAI, openai.get_state()),
                (ProviderName.ANTHROPIC, anthropic.get_state()),
            ],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        real_aclose = reporter._http.aclose

        async def send(url: str, **kwargs: object) -> MagicMock:
            if url.endswith("/breaker-reports"):
                payload = kwargs["json"]
                assert isinstance(payload, dict)
                sent_providers.append(str(payload["provider"]))
                if len(sent_providers) == 1:
                    breaker_started.set()
                    await release_breaker.wait()
            return _ok_response()

        try:
            with (
                patch.object(
                    reporter._http,
                    "post",
                    new=AsyncMock(return_value=_ok_response()),
                ) as post,
                patch.object(reporter._http, "aclose", new=AsyncMock()) as http_close,
            ):
                reporter.observe_project_id(VALID_PROJECT_ID)
                await reporter._flush_breaker_reports()
                post.reset_mock()
                openai.record_failure()
                post.side_effect = send
                reporter._start_breaker_cycle()
                await asyncio.wait_for(breaker_started.wait(), timeout=1.0)
                close_task = asyncio.create_task(reporter.close())
                await asyncio.sleep(0)

                assert not close_task.done()
                assert sent_providers == ["openai"]
                http_close.assert_not_awaited()
                release_breaker.set()
                await asyncio.wait_for(close_task, timeout=2.0)

                assert sent_providers == ["openai", "openai", "anthropic"]
                http_close.assert_awaited_once_with()
        finally:
            release_breaker.set()
            if reporter._flush_task is not None and not reporter._flush_task.done():
                await reporter._flush_task
            await real_aclose()

    @pytest.mark.asyncio
    async def test_close_deadline_cancels_active_breaker_task_and_closes_http(self) -> None:
        breaker_started = asyncio.Event()
        release_breaker = asyncio.Event()
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            breaker_snapshots=lambda: [(ProviderName.OPENAI, CircuitBreaker().get_state())],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)
        real_aclose = reporter._http.aclose

        async def send(url: str, **_kwargs: object) -> MagicMock:
            if url.endswith("/breaker-reports"):
                breaker_started.set()
                await release_breaker.wait()
            return _ok_response()

        task: asyncio.Task[None] | None = None
        try:
            with (
                patch.object(reporter._http, "post", new=AsyncMock(side_effect=send)),
                patch.object(reporter._http, "aclose", new=AsyncMock()) as http_close,
            ):
                task = reporter._start_breaker_cycle()
                assert task is not None
                await asyncio.wait_for(breaker_started.wait(), timeout=1.0)

                started_at = time.monotonic()
                await reporter.close(timeout=0.01)
                elapsed = time.monotonic() - started_at

                assert elapsed < 0.5
                assert task.done()
                http_close.assert_awaited_once_with()
        finally:
            release_breaker.set()
            if task is not None and not task.done():
                await task
            await real_aclose()

    @pytest.mark.asyncio
    async def test_next_flush_uses_current_state_and_wall_clock_only(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            breaker_snapshots=lambda: [(ProviderName.OPENAI, breaker.get_state())],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)
        before = datetime.now(UTC)

        with patch.object(
            reporter._http,
            "post",
            new_callable=AsyncMock,
            return_value=_ok_response(),
        ) as post:
            await reporter._flush_breaker_reports()
            breaker.record_failure()
            await reporter._flush_breaker_reports()

        after = datetime.now(UTC)
        payloads = [call.kwargs["json"] for call in post.call_args_list]
        assert [payload["state"] for payload in payloads] == ["closed", "open"]
        assert all(set(payload) == BREAKER_FIELDS for payload in payloads)
        for payload in payloads:
            reported_at = datetime.fromisoformat(payload["reported_at"].replace("Z", "+00:00"))
            assert before <= reported_at <= after
            assert reported_at.tzinfo is not None

        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_failure_isolated_from_confirms_metadata_and_later_providers(self) -> None:
        openai = CircuitBreaker()
        anthropic = CircuitBreaker()
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
            breaker_snapshots=lambda: [
                (ProviderName.OPENAI, openai.get_state()),
                (ProviderName.ANTHROPIC, anthropic.get_state()),
            ],
            sdk_instance_id=SDK_INSTANCE_ID,
        )
        reporter.observe_project_id(VALID_PROJECT_ID)
        reporter.report_confirm(_confirm())
        reporter.report(_event())

        async def send(url: str, **kwargs: object) -> MagicMock:
            if url.endswith("/budgets/confirm"):
                return _ok_response()
            if url.endswith("/metadata/ingest"):
                return _accepted_response({"ingested": 1, "rejected": []})
            if kwargs["json"]["provider"] == "openai":
                return _error_response()
            return _ok_response()

        with patch.object(reporter._http, "post", new=AsyncMock(side_effect=send)) as post:
            await reporter._flush_remaining()
            await reporter._flush_breaker_reports()

        breaker_calls = [
            call for call in post.call_args_list if call.args[0].endswith("/breaker-reports")
        ]
        assert [call.kwargs["json"]["provider"] for call in breaker_calls] == [
            "openai",
            "anthropic",
        ]
        assert len(reporter._confirm_queue) == 0
        assert len(reporter._queue) == 0

        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_shutdown_signal_leaves_the_single_final_flush_to_close(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        shutdown = MagicMock()
        shutdown.is_set.side_effect = [False, True]
        shutdown.wait = AsyncMock(return_value=True)
        reporter._shutdown_event = shutdown

        with patch.object(reporter, "_flush_remaining", new=AsyncMock()) as flush:
            await reporter._flush_loop()

        flush.assert_not_awaited()
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_idle_flush_tick_does_not_start_breaker_task(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )
        shutdown = MagicMock()
        shutdown.is_set.side_effect = [False, True]
        shutdown.wait = AsyncMock(side_effect=TimeoutError)
        reporter._shutdown_event = shutdown

        with (
            patch.object(reporter, "_flush_remaining", new=AsyncMock()),
            patch.object(reporter, "_breaker_reports_due", return_value=False) as due,
            patch.object(reporter, "_start_breaker_cycle") as start_cycle,
        ):
            await reporter._flush_loop()

        due.assert_called_once_with()
        start_cycle.assert_not_called()
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_close_forces_final_breaker_cycle(self) -> None:
        reporter = AsyncMetadataReporter(
            "https://api.test.solwyn.ai",
            VALID_API_KEY,
        )

        with patch.object(reporter, "_start_breaker_cycle", return_value=None) as start_cycle:
            await reporter.close()

        start_cycle.assert_called_once_with(
            during_shutdown=True,
            deadline=start_cycle.call_args.kwargs["deadline"],
            force=True,
        )
