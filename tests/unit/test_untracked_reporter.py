"""Focused tests for advisory untracked-surface reporting."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY
from pydantic import ValidationError

from solwyn import _base, _types
from solwyn._surfaces import SurfaceContext
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter

API_URL = "https://api.test.solwyn.ai"
API_KEY = "sk_test_abcdefghijklmnopqrstuvwxyz1234567890"
SDK_INSTANCE_ID = "sdk-instance-untracked-reporter"


class _UntrackedOpenAIClient:
    def post(self) -> str:
        return "posted"

    def close(self) -> None:
        return None


class _UntrackedAsyncOpenAIClient:
    async def post(self) -> str:
        return "posted"

    async def close(self) -> None:
        return None


_UntrackedOpenAIClient.__module__ = "openai._client"
_UntrackedOpenAIClient.__name__ = "OpenAI"
_UntrackedAsyncOpenAIClient.__module__ = "openai._client"
_UntrackedAsyncOpenAIClient.__name__ = "AsyncOpenAI"


EXPECTED_UNTRACKED_SURFACE_REPORT_FIELDS = {
    "provider",
    "client_shape",
    "mode",
    "surface",
    "rule_kind",
    "capability_scope",
    "posture",
    "occurrences",
    "first_seen_at",
    "last_seen_at",
    "sdk_instance_id",
    "report_id",
}


def _report_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider": "openai",
        "client_shape": "openai_sdk",
        "mode": "sync",
        "surface": "responses.create",
        "rule_kind": "unmetered_spend",
        "capability_scope": "operation",
        "posture": "warn",
        "occurrences": 3,
        "first_seen_at": datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        "last_seen_at": datetime(2026, 8, 13, 12, 1, tzinfo=UTC),
        "sdk_instance_id": "sdk-instance-1",
        "report_id": "3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
    }
    payload.update(overrides)
    return payload


def _ok_response() -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 202
    response.raise_for_status = MagicMock()
    return response


def _error_response(status_code: int) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            str(status_code),
            request=MagicMock(spec=httpx.Request),
            response=response,
        )
    )
    return response


def _quiet_sync_reporter() -> MetadataReporter:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(API_URL, API_KEY, sdk_instance_id=SDK_INSTANCE_ID)
    reporter._thread.join(timeout=2.0)
    return reporter


def _async_reporter() -> AsyncMetadataReporter:
    return AsyncMetadataReporter(API_URL, API_KEY, sdk_instance_id=SDK_INSTANCE_ID)


def _observe(
    reporter: MetadataReporter | AsyncMetadataReporter,
    surface: str = "responses.create",
    *,
    mode: str = "sync",
    n: int = 1,
    activate: bool = False,
) -> None:
    context = SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode=mode,  # type: ignore[arg-type]
    )
    for _ in range(n):
        _base._record_untracked_surface_observation(
            context=context,
            surface=surface,
            rule_kind="unmetered_spend",
            capability_scope="operation",
            posture="warn",
            notifier=(
                reporter.observe_untracked_surface
                if activate
                else reporter._untracked_state.observe
            ),
        )


@pytest.fixture(autouse=True)
def _reset_observations() -> None:
    _base._reset_unmetered_spend_warnings()
    yield
    _base._reset_unmetered_spend_warnings()


@pytest.mark.unit
def test_untracked_surface_wire_model_matches_deployed_core_contract() -> None:
    model = getattr(_types, "UntrackedSurfaceReport", None)
    assert model is not None, "SDK must mirror Core's UntrackedSurfaceReport"

    report = model(**_report_payload())
    assert set(model.model_fields) == EXPECTED_UNTRACKED_SURFACE_REPORT_FIELDS
    assert report.model_dump(mode="json") == {
        **_report_payload(),
        "first_seen_at": "2026-08-13T12:00:00Z",
        "last_seen_at": "2026-08-13T12:01:00Z",
    }
    assert get_args(_types.UntrackedClientShape) == (
        "openai_sdk",
        "native_together",
        "anthropic_sdk",
        "google_generativeai",
        "google_genai",
        "bedrock_boto3",
        "bedrock_aioboto3",
    )
    assert get_args(_types.UntrackedRuleKind) == ("unmetered_spend", "unknown")
    assert get_args(_types.UntrackedPosture) == ("warn", "allow")
    assert get_args(_types.UntrackedScope) == (
        "operation",
        "client",
        "resource",
        "raw_response",
        "arbitrary_endpoint",
    )
    assert _types.SURFACE_PATH_PATTERN == (
        r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,7}$"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"surface": "prompt content"},
        {"client_shape": "unknown_sdk"},
        {"occurrences": 0},
        {
            "first_seen_at": datetime(2026, 8, 13, 12, 1, tzinfo=UTC),
            "last_seen_at": datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        },
        {"unexpected": "forbidden"},
    ],
)
def test_untracked_surface_wire_model_rejects_contract_drift(
    overrides: dict[str, object],
) -> None:
    model = getattr(_types, "UntrackedSurfaceReport", None)
    assert model is not None, "SDK must mirror Core's UntrackedSurfaceReport"

    with pytest.raises(ValidationError):
        model(**_report_payload(**overrides))


@pytest.mark.unit
def test_untracked_surface_wire_model_normalizes_naive_timestamps_to_utc() -> None:
    report = _types.UntrackedSurfaceReport(
        **_report_payload(
            first_seen_at=datetime(2026, 8, 13, 12, 0),
            last_seen_at=datetime(2026, 8, 13, 12, 1),
        )
    )

    assert report.first_seen_at.tzinfo is UTC
    assert report.last_seen_at.tzinfo is UTC
    dumped = report.model_dump(mode="json")
    assert dumped["first_seen_at"] == "2026-08-13T12:00:00Z"
    assert dumped["last_seen_at"] == "2026-08-13T12:01:00Z"


@pytest.mark.unit
def test_untracked_surface_wire_model_normalizes_mixed_timestamps_before_comparison() -> None:
    report = _types.UntrackedSurfaceReport(
        **_report_payload(
            first_seen_at=datetime(2026, 8, 13, 12, 0),
            last_seen_at=datetime(2026, 8, 13, 12, 1, tzinfo=UTC),
        )
    )

    assert report.first_seen_at == datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    assert report.last_seen_at == datetime(2026, 8, 13, 12, 1, tzinfo=UTC)


@pytest.mark.unit
def test_untracked_surface_wire_model_rejects_mixed_invalid_range_as_validation_error() -> None:
    with pytest.raises(ValidationError, match="last_seen_at must be greater"):
        _types.UntrackedSurfaceReport(
            **_report_payload(
                first_seen_at=datetime(2026, 8, 13, 12, 1, tzinfo=UTC),
                last_seen_at=datetime(2026, 8, 13, 12, 0),
            )
        )


@pytest.mark.unit
def test_sync_first_observation_is_due_immediately_then_waits_fifteen_minutes() -> None:
    reporter = _quiet_sync_reporter()
    now = 10_000.0
    _observe(reporter)

    try:
        assert reporter._breaker_project_id is None
        with (
            patch("solwyn.reporter._monotonic", return_value=now),
            patch.object(reporter._http, "post", return_value=_ok_response()) as post,
        ):
            assert reporter._untracked_reports_due() is True
            reporter._flush_untracked_reports()

            assert post.call_count == 1
            call = post.call_args
            assert call.args[0] == f"{API_URL}/api/v1/untracked-surfaces"
            assert isinstance(call.kwargs["json"], list)
            assert len(call.kwargs["json"]) == 1
            assert call.kwargs["json"][0]["occurrences"] == 1

            _observe(reporter)
            assert reporter._untracked_reports_due() is False
            reporter._flush_untracked_reports()
            assert post.call_count == 1

            with patch("solwyn.reporter._monotonic", return_value=now + 900.0):
                assert reporter._untracked_reports_due() is True
                reporter._flush_untracked_reports()

            assert post.call_count == 2
            assert post.call_args.kwargs["json"][0]["occurrences"] == 1
    finally:
        reporter._http.close()


@pytest.mark.unit
def test_sync_failed_send_keeps_delta_and_refreshes_report_id_until_2xx() -> None:
    reporter = _quiet_sync_reporter()
    _observe(reporter, n=2)

    try:
        with (
            patch("solwyn.reporter._monotonic", return_value=100.0),
            patch.object(
                reporter._http,
                "post",
                side_effect=[_error_response(500), _ok_response()],
            ) as post,
        ):
            reporter._flush_untracked_reports()
            first_payload = post.call_args.kwargs["json"]
            assert reporter._untracked_reports_due() is False
            assert reporter.dropped_counts == {}

            _observe(reporter)
            with patch("solwyn.reporter._monotonic", return_value=1_000.0):
                assert reporter._untracked_reports_due() is True
                reporter._flush_untracked_reports()
            second_payload = post.call_args.kwargs["json"]

            assert first_payload[0]["occurrences"] == 2
            assert second_payload[0]["occurrences"] == 3
            assert first_payload[0]["report_id"] != second_payload[0]["report_id"]
            assert reporter._untracked_reports_due() is False
            assert reporter.dropped_counts == {}
    finally:
        reporter._http.close()


@pytest.mark.unit
def test_sync_observation_during_successful_send_remains_in_next_delta() -> None:
    reporter = _quiet_sync_reporter()
    _observe(reporter)

    def send(_url: str, **_kwargs: object) -> MagicMock:
        _observe(reporter)
        return _ok_response()

    try:
        with (
            patch("solwyn.reporter._monotonic", return_value=100.0),
            patch.object(reporter._http, "post", side_effect=send) as post,
        ):
            reporter._flush_untracked_reports()
            with patch("solwyn.reporter._monotonic", return_value=1_000.0):
                reporter._flush_untracked_reports()

        assert post.call_args_list[0].kwargs["json"][0]["occurrences"] == 1
        assert post.call_args_list[1].kwargs["json"][0]["occurrences"] == 1
    finally:
        reporter._http.close()


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [404, 500])
def test_sync_http_errors_drop_silently_without_spend_drop_accounting(status_code: int) -> None:
    reporter = _quiet_sync_reporter()
    _observe(reporter)

    try:
        with patch.object(
            reporter._http,
            "post",
            return_value=_error_response(status_code),
        ):
            reporter._flush_untracked_reports()

        assert reporter.dropped_counts == {}
    finally:
        reporter._http.close()


@pytest.mark.unit
def test_sync_batches_untracked_reports_at_one_hundred() -> None:
    reporter = _quiet_sync_reporter()
    for index in range(101):
        _observe(reporter, surface=f"surface_{index}")

    try:
        with patch.object(reporter._http, "post", return_value=_ok_response()) as post:
            reporter._flush_untracked_reports()

        assert [len(call.kwargs["json"]) for call in post.call_args_list] == [100, 1]
    finally:
        reporter._http.close()


@pytest.mark.unit
def test_sync_mixed_batch_outcome_advances_only_successful_batch() -> None:
    reporter = _quiet_sync_reporter()
    for index in range(101):
        _observe(reporter, surface=f"surface_{index}")

    try:
        with (
            patch("solwyn.reporter._monotonic", return_value=100.0),
            patch.object(
                reporter._http,
                "post",
                side_effect=[_ok_response(), _error_response(500), _ok_response()],
            ) as post,
        ):
            reporter._flush_untracked_reports()
            failed_payload = post.call_args_list[1].kwargs["json"]
            with patch("solwyn.reporter._monotonic", return_value=1_000.0):
                reporter._flush_untracked_reports()

        assert [len(call.kwargs["json"]) for call in post.call_args_list] == [100, 1, 1]
        retry_payload = post.call_args_list[2].kwargs["json"]
        assert retry_payload[0]["surface"] == failed_payload[0]["surface"]
        assert retry_payload[0]["occurrences"] == 1
        assert retry_payload[0]["report_id"] != failed_payload[0]["report_id"]
    finally:
        reporter._http.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "second_api_key",
    [API_KEY, "sk_test_second_project_abcdefghijklmnopqrstuvwxyz"],
    ids=["same-project-key", "different-project-key"],
)
def test_sync_reporters_only_drain_their_originating_observations(second_api_key: str) -> None:
    first = _quiet_sync_reporter()
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        second = MetadataReporter(API_URL, second_api_key, sdk_instance_id="sdk-instance-second")
    second._thread.join(timeout=2.0)
    _observe(first, n=2)

    try:
        with (
            patch.object(first._http, "post", return_value=_ok_response()) as first_post,
            patch.object(second._http, "post", return_value=_ok_response()) as second_post,
        ):
            first._flush_untracked_reports()
            second._flush_untracked_reports()

        first_post.assert_called_once()
        assert first_post.call_args.kwargs["json"][0]["occurrences"] == 2
        second_post.assert_not_called()
        assert (
            _base._untracked_surface_observations[
                ("openai", "openai_sdk", "sync", "responses.create")
            ]["occurrences"]
            == 2
        )
    finally:
        first._http.close()
        second._http.close()


@pytest.mark.unit
@pytest.mark.parametrize("captured_before_fork", [False, True], ids=["unsent", "captured"])
def test_sync_fork_reset_discards_only_inherited_advisory_observations(
    captured_before_fork: bool,
) -> None:
    parent = _quiet_sync_reporter()
    child = _quiet_sync_reporter()
    parent_http = parent._http
    inherited_child_http = child._http
    _observe(parent, n=2)
    _observe(child, n=2)
    child_state = child._untracked_state
    assert child_state is not None
    inherited_reports = child._build_untracked_reports() if captured_before_fork else []
    old_lock = child_state.lock
    if captured_before_fork:
        child._mark_untracked_reports_attempted(inherited_reports)

    child._reset_after_fork_in_child()

    try:
        assert child._untracked_state is child_state
        assert child_state.lock is not old_lock
        assert child_state.sdk_instance_id == SDK_INSTANCE_ID
        assert [built.report.occurrences for built in parent._build_untracked_reports()] == [2]
        assert child._build_untracked_reports() == []

        _observe(child)
        if captured_before_fork:
            # A pre-fork worker cannot continue after a real fork. These calls
            # deterministically model a stale captured completion and must be inert.
            child._mark_untracked_reports_attempted(inherited_reports)
            child._mark_untracked_reports_sent(inherited_reports)
        post_fork_reports = child._build_untracked_reports()

        assert [built.report.occurrences for built in post_fork_reports] == [1]
        if captured_before_fork:
            assert post_fork_reports[0].report.report_id != inherited_reports[0].report.report_id
        child._mark_untracked_reports_sent(post_fork_reports)
        assert child._build_untracked_reports() == []
    finally:
        parent._shutdown.set()
        child._shutdown.set()
        parent_http.close()
        inherited_child_http.close()
        child._http.close()


@pytest.mark.unit
@pytest.mark.parametrize("captured_before_fork", [False, True], ids=["unsent", "captured"])
@pytest.mark.asyncio
async def test_async_fork_reset_discards_only_inherited_advisory_observations(
    captured_before_fork: bool,
) -> None:
    parent = _async_reporter()
    child = _async_reporter()
    parent_http = parent._http
    inherited_child_http = child._http
    _observe(parent, mode="async", n=2)
    _observe(child, mode="async", n=2)
    child_state = child._untracked_state
    assert child_state is not None
    inherited_reports = child._build_untracked_reports() if captured_before_fork else []
    old_lock = child_state.lock
    if captured_before_fork:
        child._mark_untracked_reports_attempted(inherited_reports)

    child._reset_after_fork_in_child()

    try:
        assert child._untracked_state is child_state
        assert child_state.lock is not old_lock
        assert child_state.sdk_instance_id == SDK_INSTANCE_ID
        assert [built.report.occurrences for built in parent._build_untracked_reports()] == [2]
        assert child._build_untracked_reports() == []

        _observe(child, mode="async")
        if captured_before_fork:
            child._mark_untracked_reports_attempted(inherited_reports)
            child._mark_untracked_reports_sent(inherited_reports)
        post_fork_reports = child._build_untracked_reports()

        assert [built.report.occurrences for built in post_fork_reports] == [1]
        if captured_before_fork:
            assert post_fork_reports[0].report.report_id != inherited_reports[0].report.report_id
        child._mark_untracked_reports_sent(post_fork_reports)
        assert child._build_untracked_reports() == []
    finally:
        await parent_http.aclose()
        await inherited_child_http.aclose()
        await child._http.aclose()


@pytest.mark.unit
def test_global_registry_saturation_does_not_suppress_origin_reporter_delivery() -> None:
    reporter = _quiet_sync_reporter()
    context = SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )
    for index in range(_base._UNTRACKED_SURFACE_LIMIT):
        _base._record_untracked_surface_observation(
            context=context,
            surface=f"disabled_{index}",
            rule_kind="unknown",
            capability_scope=None,
            posture="allow",
            notifier=None,
        )

    try:
        with patch.object(reporter, "_notify_untracked_observation") as wake:
            for _ in range(2):
                _base._record_untracked_surface_observation(
                    context=context,
                    surface="responses.create",
                    rule_kind="unmetered_spend",
                    capability_scope="operation",
                    posture="allow",
                    notifier=reporter.observe_untracked_surface,
                )

        assert len(_base._untracked_surface_observations) == _base._UNTRACKED_SURFACE_LIMIT
        assert ("openai", "openai_sdk", "sync", "responses.create") not in (
            _base._untracked_surface_observations
        )
        assert wake.call_count == 2
        with patch.object(reporter._http, "post", return_value=_ok_response()) as post:
            reporter._flush_untracked_reports()
        post.assert_called_once()
        assert post.call_args.kwargs["json"][0]["occurrences"] == 2
    finally:
        reporter._http.close()


@pytest.mark.unit
def test_origin_reporter_ledger_keeps_its_own_cardinality_bound() -> None:
    reporter = _quiet_sync_reporter()
    state = reporter._untracked_state
    assert state is not None
    context = SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )
    seen_at = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    for index in range(_base._UNTRACKED_SURFACE_LIMIT):
        state.observe(
            context=context,
            surface=f"surface_{index}",
            rule_kind="unknown",
            capability_scope=None,
            posture="allow",
            seen_at=seen_at,
        )
    state.observe(
        context=context,
        surface="overflow",
        rule_kind="unknown",
        capability_scope=None,
        posture="allow",
        seen_at=seen_at,
    )
    state.observe(
        context=context,
        surface="surface_0",
        rule_kind="unknown",
        capability_scope=None,
        posture="allow",
        seen_at=seen_at,
    )

    try:
        assert len(state.observations) == _base._UNTRACKED_SURFACE_LIMIT
        assert ("openai", "openai_sdk", "sync", "overflow") not in state.observations
        assert state.observations[("openai", "openai_sdk", "sync", "surface_0")]["occurrences"] == 2
    finally:
        reporter._http.close()


@pytest.mark.unit
def test_advisory_notifier_failure_never_escapes_the_provider_call_path() -> None:
    context = SurfaceContext(
        provider="openai",
        dialect="openai",
        client_shape="openai_sdk",
        mode="sync",
    )

    def fail_notifier(**_kwargs: object) -> None:
        raise RuntimeError("advisory notifier failed")

    status = _base._record_untracked_surface_observation(
        context=context,
        surface="responses.create",
        rule_kind="unmetered_spend",
        capability_scope="operation",
        posture="allow",
        notifier=fail_notifier,
    )

    assert status == "silent"
    assert (
        _base._untracked_surface_observations[("openai", "openai_sdk", "sync", "responses.create")][
            "occurrences"
        ]
        == 1
    )


@pytest.mark.unit
def test_sync_untracked_only_call_wakes_reporter_without_budget_bootstrap() -> None:
    client = Solwyn(
        _UntrackedOpenAIClient(),
        api_key=VALID_API_KEY,
        on_unmetered="allow",
        reporter_flush_interval=3_600.0,
    )
    sent = threading.Event()
    caller_thread = threading.current_thread()

    def send(url: str, **_kwargs: object) -> MagicMock:
        assert url == f"{client._config.api_url}/api/v1/untracked-surfaces"
        assert threading.current_thread() is not caller_thread
        sent.set()
        return _ok_response()

    try:
        with (
            patch.object(client._reporter._http, "post", side_effect=send),
            patch.object(client._budget._http, "post") as budget_post,
        ):
            assert client.post() == "posted"
            assert sent.wait(timeout=1.0), "origin reporter was not woken by observation"
            budget_post.assert_not_called()
    finally:
        client.close()


@pytest.mark.unit
def test_sync_disabled_reporting_stays_local_through_periodic_flush_and_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = Solwyn(
        _UntrackedOpenAIClient(),
        api_key=VALID_API_KEY,
        on_unmetered="warn",
        report_untracked_surfaces=False,
        reporter_flush_interval=0.01,
    )

    with patch.object(client._reporter._http, "post", return_value=_ok_response()) as post:
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            assert client.post() == "posted"
            assert client.post() == "posted"
        threading.Event().wait(0.03)
        client.close()

    key = ("openai", "openai_sdk", "sync", "post")
    assert _base._untracked_surface_observations[key]["occurrences"] == 2
    assert len([record for record in caplog.records if record.name == "solwyn._base"]) == 1
    assert client._untracked_observation_notifier is None
    assert client._reporter._untracked_state is None
    assert client._reporter._untracked_worker is None
    post.assert_not_called()


@pytest.mark.unit
def test_sync_untracked_observation_restarts_and_wakes_reporter_after_fork() -> None:
    reporter = _quiet_sync_reporter()
    parent_http = reporter._http
    reporter._reset_after_fork_in_child()
    replacement_thread = MagicMock(spec=threading.Thread)

    try:
        with (
            patch.object(
                reporter,
                "_launch_thread",
                return_value=replacement_thread,
            ) as launch,
            patch.object(reporter, "_start_untracked_cycle") as start_cycle,
        ):
            _observe(reporter, activate=True)

        launch.assert_called_once_with()
        start_cycle.assert_called_once_with()
        assert reporter._thread is replacement_thread
        assert reporter._needs_thread_restart is False
    finally:
        reporter._shutdown.set()
        reporter._http.close()
        parent_http.close()


@pytest.mark.unit
def test_sync_observation_coalesced_behind_active_cycle_is_woken_after_completion() -> None:
    reporter = _quiet_sync_reporter()
    started = threading.Event()
    release = threading.Event()
    _observe(reporter)

    def send(_url: str, **_kwargs: object) -> MagicMock:
        started.set()
        release.wait(timeout=1.0)
        return _ok_response()

    try:
        with patch.object(reporter._http, "post", side_effect=send):
            worker = reporter._start_untracked_cycle()
            assert worker is not None
            assert started.wait(timeout=1.0)

            _observe(reporter, surface="embeddings.create", activate=True)
            assert reporter._start_untracked_cycle() is worker
            release.set()
            worker.join(timeout=1.0)
            continuation = reporter._untracked_worker
            if continuation is not None:
                continuation.join(timeout=1.0)

        assert reporter._untracked_reports_due() is False
    finally:
        release.set()
        reporter._shutdown.set()
        reporter._http.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_first_observation_and_failed_delta_match_sync() -> None:
    reporter = _async_reporter()
    _observe(reporter, mode="async", n=2)

    assert reporter._breaker_project_id is None
    with (
        patch("solwyn.reporter._monotonic", return_value=500.0),
        patch.object(
            reporter._http,
            "post",
            new=AsyncMock(side_effect=[_error_response(404), _ok_response()]),
        ) as post,
    ):
        assert reporter._untracked_reports_due() is True
        await reporter._flush_untracked_reports()
        assert reporter._untracked_reports_due() is False
        _observe(reporter, mode="async")
        with patch("solwyn.reporter._monotonic", return_value=1_400.0):
            assert reporter._untracked_reports_due() is True
            await reporter._flush_untracked_reports()

        assert post.call_count == 2
        assert post.call_args_list[0].kwargs["json"][0]["occurrences"] == 2
        assert post.call_args_list[1].kwargs["json"][0]["occurrences"] == 3
        assert reporter._untracked_reports_due() is False
        assert reporter.dropped_counts == {}

    await reporter._http.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_batches_untracked_reports_at_one_hundred() -> None:
    reporter = _async_reporter()
    with patch.object(reporter, "_notify_untracked_observation"):
        for index in range(101):
            _observe(reporter, surface=f"surface_{index}", mode="async")

    with patch.object(
        reporter._http,
        "post",
        new=AsyncMock(return_value=_ok_response()),
    ) as post:
        await reporter._flush_untracked_reports()

    assert [len(call.kwargs["json"]) for call in post.call_args_list] == [100, 1]
    await reporter._http.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_observation_coalesced_behind_active_cycle_is_woken_after_completion() -> None:
    reporter = _async_reporter()
    reporter._shutdown_event = asyncio.Event()
    started = asyncio.Event()
    release = asyncio.Event()
    _observe(reporter, mode="async")

    async def send(_url: str, **_kwargs: object) -> MagicMock:
        started.set()
        await release.wait()
        return _ok_response()

    try:
        with patch.object(reporter._http, "post", new=AsyncMock(side_effect=send)):
            task = reporter._start_untracked_cycle()
            assert task is not None
            await asyncio.wait_for(started.wait(), timeout=1.0)

            _observe(
                reporter,
                surface="embeddings.create",
                mode="async",
                activate=True,
            )
            assert reporter._start_untracked_cycle() is task
            release.set()
            await asyncio.wait_for(task, timeout=1.0)
            await asyncio.sleep(0)
            continuation = reporter._untracked_task
            assert continuation is not None
            await asyncio.wait_for(continuation, timeout=1.0)

        assert reporter._untracked_reports_due() is False
    finally:
        release.set()
        await reporter._http.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_untracked_only_call_auto_starts_and_wakes_reporter() -> None:
    client = AsyncSolwyn(
        _UntrackedAsyncOpenAIClient(),
        api_key=VALID_API_KEY,
        on_unmetered="allow",
        reporter_flush_interval=3_600.0,
    )
    sent = asyncio.Event()

    async def send(url: str, **_kwargs: object) -> MagicMock:
        assert url == f"{client._config.api_url}/api/v1/untracked-surfaces"
        sent.set()
        return _ok_response()

    try:
        with (
            patch.object(client._reporter._http, "post", new=AsyncMock(side_effect=send)),
            patch.object(client._budget._http, "post", new=AsyncMock()) as budget_post,
        ):
            assert await client.post() == "posted"
            await asyncio.wait_for(sent.wait(), timeout=1.0)
            assert client._reporter._flush_task is not None
            budget_post.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_disabled_reporting_stays_local_through_periodic_flush_and_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = AsyncSolwyn(
        _UntrackedAsyncOpenAIClient(),
        api_key=VALID_API_KEY,
        on_unmetered="warn",
        report_untracked_surfaces=False,
        reporter_flush_interval=0.01,
    )
    client._reporter.start()

    with patch.object(
        client._reporter._http,
        "post",
        new=AsyncMock(return_value=_ok_response()),
    ) as post:
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            assert await client.post() == "posted"
            assert await client.post() == "posted"
        await asyncio.sleep(0.03)
        await client.close()

    key = ("openai", "openai_sdk", "async", "post")
    assert _base._untracked_surface_observations[key]["occurrences"] == 2
    assert len([record for record in caplog.records if record.name == "solwyn._base"]) == 1
    assert client._untracked_observation_notifier is None
    assert client._reporter._untracked_state is None
    assert client._reporter._untracked_task is None
    post.assert_not_awaited()


@pytest.mark.unit
def test_sync_close_initiates_final_due_untracked_cycle_before_transport_close() -> None:
    reporter = _quiet_sync_reporter()
    _observe(reporter)

    with patch.object(reporter._http, "post", return_value=_ok_response()) as post:
        reporter.close(timeout=1.0)

    post.assert_called_once()
    assert post.call_args.args[0] == f"{API_URL}/api/v1/untracked-surfaces"
    assert reporter.dropped_counts == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_close_initiates_final_due_untracked_cycle_before_transport_close() -> None:
    reporter = _async_reporter()
    with patch.object(reporter, "_notify_untracked_observation"):
        _observe(reporter, mode="async")

    with patch.object(
        reporter._http,
        "post",
        new=AsyncMock(return_value=_ok_response()),
    ) as post:
        await reporter.close(timeout=1.0)

    post.assert_awaited_once()
    assert post.call_args.args[0] == f"{API_URL}/api/v1/untracked-surfaces"
    assert reporter.dropped_counts == {}


@pytest.mark.unit
def test_sync_flush_loop_starts_untracked_cycle_only_when_due() -> None:
    reporter = _quiet_sync_reporter()
    shutdown = MagicMock(spec=__import__("threading").Event)
    shutdown.is_set.side_effect = [False, True]
    shutdown.wait.return_value = False
    reporter._shutdown = shutdown

    try:
        with (
            patch.object(reporter, "_flush_remaining"),
            patch.object(reporter, "_breaker_reports_due", return_value=False),
            patch.object(reporter, "_untracked_reports_due", return_value=True) as due,
            patch.object(reporter, "_start_untracked_cycle") as start_cycle,
        ):
            reporter._flush_loop()

        due.assert_called_once_with()
        start_cycle.assert_called_once_with()
    finally:
        reporter._http.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_flush_loop_starts_untracked_cycle_only_when_due() -> None:
    reporter = _async_reporter()
    reporter._shutdown_event = asyncio.Event()
    reporter.flush_interval = 0.0

    async def flush_once() -> None:
        reporter._shutdown_event.set()

    with (
        patch.object(reporter, "_flush_remaining", side_effect=flush_once),
        patch.object(reporter, "_breaker_reports_due", return_value=False),
        patch.object(reporter, "_untracked_reports_due", return_value=True) as due,
        patch.object(reporter, "_start_untracked_cycle") as start_cycle,
    ):
        await reporter._flush_loop()

    due.assert_called_once_with()
    start_cycle.assert_called_once_with()
    await reporter._http.aclose()
