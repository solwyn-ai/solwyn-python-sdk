"""Read-only API keys are a persistent configuration refusal, never fail-open.

A runtime ``403 read_only_key`` blocks the call that met it and every later
call without network, whatever ``fail_open`` says. Background items that meet
it are dropped terminally (no retry) and the configuration error logs once.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, call_uuid, validated_binding

from solwyn._api_key_errors import classify_exception
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.exceptions import ApiKeyRefusedError
from solwyn.reporter import (
    AsyncMetadataReporter,
    MetadataReporter,
    _PendingConfirm,
    _PendingEvent,
)


def _read_only_response() -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 403
    response.json.return_value = {
        "detail": {
            "code": "read_only_key",
            "message": "This API key is read-only and cannot write project data",
        }
    }
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "forbidden",
        request=MagicMock(spec=httpx.Request),
        response=response,
    )
    return response


def _event() -> MetadataEvent:
    return MetadataEvent(
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        input_tokens=10,
        output_tokens=5,
        latency_ms=100,
        status="success",
        is_model_fallback=False,
        sdk_instance_id="read-only-test",
        timestamp=datetime.now(UTC),
        call_id=call_uuid("call-read-only"),
    )


def _confirm() -> BudgetConfirmRequest:
    return BudgetConfirmRequest(
        reservation_id="res-read-only",
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        call_id=call_uuid("call-read-only"),
        token_details=TokenDetails(input_tokens=10, output_tokens=5),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        # Every 401 is a key refusal, whatever its body says.
        (401, {"detail": {"code": "read_only_key"}}, "api_key_invalid"),
        # A 403 classifies only on a recognised detail.code.
        (403, {"code": "read_only_key"}, None),
        (403, {"detail": {"code": "permission_denied"}}, None),
        (403, {"detail": {"code": "read_only_key"}}, "read_only_key"),
    ],
)
def test_read_only_classification_near_misses(
    status_code: int,
    payload: object,
    expected: str | None,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = payload
    error = httpx.HTTPStatusError(
        "request failed",
        request=MagicMock(spec=httpx.Request),
        response=response,
    )

    refusal = classify_exception(error)

    assert (refusal.code if refusal is not None else None) == expected


@pytest.mark.unit
def test_sync_budget_read_only_errors_block_persistently_and_log_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    enforcer = BudgetEnforcer(
        "https://api.test.solwyn.ai", validated_binding(VALID_API_KEY), fail_open=True
    )

    with (
        patch.object(enforcer._http, "post", return_value=_read_only_response()) as post,
        caplog.at_level("WARNING"),
    ):
        with pytest.raises(ApiKeyRefusedError) as first:
            enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
            )
        with pytest.raises(ApiKeyRefusedError):
            enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
            )

    # fail_open never applies to a refusal; the second call is blocked at the
    # gate without a request.
    assert first.value.code == "read_only_key"
    assert first.value.field == "api_key"
    assert post.call_count == 1
    assert caplog.text.count("solwyn.configuration_error.read_only_key") == 1
    assert "Cloud API budget check failed" not in caplog.text
    enforcer.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_budget_read_only_errors_block_persistently_and_log_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    enforcer = AsyncBudgetEnforcer(
        "https://api.test.solwyn.ai",
        validated_binding(VALID_API_KEY),
        fail_open=True,
    )
    enforcer._http.post = AsyncMock(return_value=_read_only_response())

    with caplog.at_level("WARNING"):
        for _ in range(2):
            with pytest.raises(ApiKeyRefusedError):
                await enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                )

    assert enforcer._http.post.await_count == 1
    assert caplog.text.count("solwyn.configuration_error.read_only_key") == 1
    assert "Cloud API budget check failed" not in caplog.text
    await enforcer.close()


@pytest.mark.unit
def test_sync_reporter_read_only_batches_are_terminal_and_log_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(
            "https://api.test.solwyn.ai",
            validated_binding(VALID_API_KEY),
            batch_size=1,
        )
    reporter._thread.join(timeout=2)
    reporter._confirm_queue.append(_PendingConfirm(_confirm()))
    reporter._queue.append(_PendingEvent(_event()))

    with (
        patch.object(reporter._http, "post", return_value=_read_only_response()) as post,
        caplog.at_level("WARNING"),
    ):
        reporter._flush_remaining()

    # The first refusal stops all traffic: everything after it is dropped
    # locally, unsent.
    assert post.call_count == 1
    assert len(reporter._confirm_queue) == 0
    assert len(reporter._queue) == 0
    assert reporter._consecutive_confirm_failures == 0
    assert caplog.text.count("solwyn.configuration_error.read_only_key") == 1
    assert "reporter.confirm_send_failed" not in caplog.text
    assert "Failed to send metadata batch" not in caplog.text
    reporter._http.close()


@pytest.mark.unit
def test_sync_breaker_reports_read_only_errors_log_configuration_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    openai = CircuitBreaker()
    anthropic = CircuitBreaker()
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(
            "https://api.test.solwyn.ai",
            validated_binding(VALID_API_KEY),
            breaker_snapshots=lambda: [
                (ProviderName.OPENAI, openai.get_state()),
                (ProviderName.ANTHROPIC, anthropic.get_state()),
            ],
            sdk_instance_id="read-only-test",
        )
    reporter._thread.join(timeout=2)

    with (
        patch.object(reporter._http, "post", return_value=_read_only_response()) as post,
        caplog.at_level("WARNING"),
    ):
        reporter._flush_breaker_reports()
        reporter._flush_breaker_reports()

    # A read-only key can never post a breaker report: the first 403 ends the
    # cycle (one POST per cycle, not one per provider) and later cycles stay
    # silent after the single one-time configuration error.
    # The first refusal stops all traffic: everything after it is dropped
    # locally, unsent.
    assert post.call_count == 1
    assert caplog.text.count("solwyn.configuration_error.read_only_key") == 1
    assert "reporter.breaker_send_failed" not in caplog.text
    reporter._http.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_breaker_reports_read_only_errors_log_configuration_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    openai = CircuitBreaker()
    anthropic = CircuitBreaker()
    reporter = AsyncMetadataReporter(
        "https://api.test.solwyn.ai",
        validated_binding(VALID_API_KEY),
        breaker_snapshots=lambda: [
            (ProviderName.OPENAI, openai.get_state()),
            (ProviderName.ANTHROPIC, anthropic.get_state()),
        ],
        sdk_instance_id="read-only-test",
    )
    reporter._http.post = AsyncMock(return_value=_read_only_response())

    with caplog.at_level("WARNING"):
        await reporter._flush_breaker_reports()
        await reporter._flush_breaker_reports()

    # The first refusal stops all traffic: everything after it is dropped
    # locally, unsent.
    assert reporter._http.post.await_count == 1
    assert caplog.text.count("solwyn.configuration_error.read_only_key") == 1
    assert "reporter.breaker_send_failed" not in caplog.text
    await reporter._http.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_reporter_read_only_batches_are_terminal_and_log_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reporter = AsyncMetadataReporter(
        "https://api.test.solwyn.ai",
        validated_binding(VALID_API_KEY),
        batch_size=1,
    )
    reporter._confirm_queue.append(_PendingConfirm(_confirm()))
    reporter._queue.append(_PendingEvent(_event()))
    reporter._http.post = AsyncMock(return_value=_read_only_response())

    with caplog.at_level("WARNING"):
        await reporter._flush_remaining()

    # The first refusal stops all traffic: everything after it is dropped
    # locally, unsent.
    assert reporter._http.post.await_count == 1
    assert len(reporter._confirm_queue) == 0
    assert len(reporter._queue) == 0
    assert reporter._consecutive_confirm_failures == 0
    assert caplog.text.count("solwyn.configuration_error.read_only_key") == 1
    assert "reporter.confirm_send_failed" not in caplog.text
    assert "Failed to send metadata batch" not in caplog.text
    await reporter._http.aclose()
