"""Same-provider retry on a 429 with a usable Retry-After.

When ``same_provider_retries > 0`` and a candidate returns HTTP 429 carrying a
Retry-After the provider asked us to honor -- delta-seconds or HTTP-date -- that
fits the remaining chain ``Deadline``, the dispatch loop sleeps the delay and
re-attempts the SAME provider before failing over cross-provider. The retried
429 records NO breaker verdict until the retry itself resolves (success closes /
neutralizes the probe; exhaustion records exactly one failure then fails over).
Default ``same_provider_retries=0`` keeps today's immediate-failover behavior.

Clients are duck-typed MagicMocks (see ``test_failover_routing``); the 429
carries its Retry-After on ``.response.headers``, mirroring the OpenAI/Anthropic
exception shape the parser reads without importing any provider SDK.
``time.sleep`` / ``asyncio.sleep`` are patched so tests assert the honored delay
with no real wait.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._types import CircuitState
from solwyn.client import AsyncSolwyn, Solwyn

# ── duck-typed fakes ─────────────────────────────────────────────────────


class _Status429RetryAfter(Exception):
    """A 429 carrying a Retry-After header on ``.response.headers``.

    Mirrors the OpenAI/Anthropic exception shape the parser reads: an HTTP
    status on ``status_code`` plus an httpx-style ``response.headers`` mapping.
    ``retry_after=None`` omits the header (a bare 429 -> no retry).
    """

    def __init__(self, retry_after: str | None = "2") -> None:
        super().__init__("rate limited")
        self.status_code = 429
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["retry-after"] = retry_after
        self.response = SimpleNamespace(headers=headers)


class _Status(Exception):
    """Duck-typed transport error carrying only an HTTP ``status_code``.

    429 -> FAILOVER, 4xx -> FAIL_FAST, 5xx -> POST_SEND_AMBIGUOUS. Carries no
    Retry-After, so a 429 _Status never qualifies for the same-provider retry.
    """

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    return client


def _anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"
    client.with_options.return_value = client
    return client


def _openai_response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok from gpt", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_response() -> SimpleNamespace:
    block = SimpleNamespace(type="text", text="ok from claude")
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _allow_budget() -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=None,
        project_id=VALID_PROJECT_ID,
        price_hints=None,
    )


def _make_solwyn(client: object, **overrides: object) -> Solwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    solwyn._reporter.report = MagicMock()
    return solwyn


def _close(solwyn: Solwyn) -> None:
    solwyn._reporter._http.close()
    solwyn._budget._http.close()


class _FixedDeadline:
    """A ``Deadline`` stand-in with a fixed remaining budget that never expires.

    Lets a test prove that a Retry-After larger than the remaining chain budget
    is NOT honored, with zero dependence on the wall clock.
    """

    def __init__(self, total: float) -> None:
        pass

    def expired(self) -> bool:
        return False

    def remaining(self) -> float:
        return 5.0


_PLAIN_REQUEST = {
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "hi"}],
}


# ── sync ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSameProviderRetryOn429:
    def test_retry_same_provider_then_succeeds(self) -> None:
        # 429 with a 2s Retry-After, then the SAME provider succeeds on retry.
        # The fallback is never touched and the retried-then-succeeded 429
        # records NO breaker failure.
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("2"), success]
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.call_count == 2
        anthropic.messages.create.assert_not_called()
        sleep.assert_called_once_with(2.0)
        assert openai_cb.failure_count == 0
        assert openai_cb.state == CircuitState.CLOSED

        _close(solwyn)

    def test_retry_exhausted_falls_over(self) -> None:
        # 429, retry (still 429), single retry exhausted -> fail over. Exactly
        # ONE breaker failure for the provider (the terminal 429), not per attempt.
        openai = _openai_client()
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create.return_value = anthropic_resp

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")
        anthropic_cb = solwyn._get_circuit_breaker("anthropic")
        attempts = 0

        with (
            patch.object(
                openai_cb, "record_failure", wraps=openai_cb.record_failure
            ) as record_failure,
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):

            def openai_create(*_args: object, **_kwargs: object) -> object:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise _Status429RetryAfter("0")
                assert record_failure.call_count == 0
                raise _Status429RetryAfter("0")

            openai.chat.completions.create.side_effect = openai_create
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert attempts == 2
        assert openai.chat.completions.create.call_count == 2
        anthropic.messages.create.assert_called_once()
        assert result is not anthropic_resp  # normalized back to the OpenAI dialect
        assert result.choices[0].message.content == "ok from claude"
        sleep.assert_called_once_with(0.0)
        record_failure.assert_called_once_with()
        assert openai_cb.failure_count == 1
        assert anthropic_cb.failure_count == 0

        _close(solwyn)

    def test_retry_after_exceeds_deadline_skips_retry(self) -> None:
        # Retry-After (30s) exceeds the remaining chain budget (5s): no retry,
        # no sleep -- fail over immediately.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status429RetryAfter("30")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.Deadline", _FixedDeadline),
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert openai.chat.completions.create.call_count == 1
        anthropic.messages.create.assert_called_once()
        sleep.assert_not_called()
        assert openai_cb.failure_count == 1
        assert result is not None

        _close(solwyn)

    def test_default_zero_fails_over_immediately(self) -> None:
        # Default same_provider_retries=0: a 429 -- even carrying a Retry-After --
        # fails over immediately, exactly as today. No retry, no sleep.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status429RetryAfter("2")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert openai.chat.completions.create.call_count == 1
        anthropic.messages.create.assert_called_once()
        sleep.assert_not_called()
        assert openai_cb.failure_count == 1

        _close(solwyn)

    def test_half_open_probe_not_stranded_on_retry(self) -> None:
        # A HALF_OPEN probe that 429s-then-succeeds holds its single probe slot
        # across the sleep (no neutral release, no mid-retry verdict), then closes
        # the breaker on the retry's success -- the slot is freed, not stranded.
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("0"), success]

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )
        openai_cb = solwyn._get_circuit_breaker("openai")
        openai_cb.record_failure()  # -> OPEN; recovery_timeout=0 so admit() probes
        assert openai_cb.state == CircuitState.OPEN

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.call_count == 2
        sleep.assert_called_once_with(0.0)
        assert openai_cb.state == CircuitState.CLOSED
        assert openai_cb._half_open_probe_active is False

        _close(solwyn)

    def test_intermediate_retry_emits_no_error_event(self) -> None:
        # Decision 2: a retried (intermediate) 429 emits NO error event -- only the
        # terminal outcome reports. Pins the unchanged Cloud-API wire contract.
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("0"), success]

        solwyn = _make_solwyn(openai, model="gpt-5.5", same_provider_retries=1)
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with (
            patch("solwyn.client.time.sleep"),
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert [e for e in events if e.status.value == "error"] == []
        assert len([e for e in events if e.status.value == "success"]) == 1

        _close(solwyn)

    def test_exhausted_emits_single_terminal_error_event(self) -> None:
        # One error event for the provider's terminal 429 (NOT one per retry), plus
        # the fallback's success event.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = [
            _Status429RetryAfter("0"),
            _Status429RetryAfter("0"),
        ]
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with (
            patch("solwyn.client.time.sleep"),
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        error_events = [e for e in events if e.status.value == "error"]
        success_events = [e for e in events if e.status.value == "success"]
        assert len(error_events) == 1
        assert error_events[0].provider == "openai"
        assert len(success_events) == 1
        assert success_events[0].provider == "anthropic"

        _close(solwyn)

    def test_two_retries_then_succeeds(self) -> None:
        # same_provider_retries=2 allows two retries (3 attempts); success on the
        # third neutralizes the breaker. Guards a boolean/off-by-one counter.
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create.side_effect = [
            _Status429RetryAfter("0"),
            _Status429RetryAfter("0"),
            success,
        ]

        solwyn = _make_solwyn(openai, model="gpt-5.5", same_provider_retries=2)
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.call_count == 3
        assert sleep.call_count == 2
        assert openai_cb.failure_count == 0

        _close(solwyn)

    def test_two_retries_exhausted_falls_over(self) -> None:
        # Two retries exhausted (3 attempts), then a SINGLE failure + failover.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("0")] * 3
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=2,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert openai.chat.completions.create.call_count == 3
        assert sleep.call_count == 2
        anthropic.messages.create.assert_called_once()
        assert openai_cb.failure_count == 1

        _close(solwyn)

    def test_retry_after_margin_boundary_skips_retry(self) -> None:
        # remaining=5.0; Retry-After 5 + _MIN_HOP_TIMEOUT(1.0) = 6.0 > 5.0 -- the
        # re-attempt would not fit, so the retry is skipped. Pins the margin term.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status429RetryAfter("5")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch("solwyn.client.Deadline", _FixedDeadline),
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert openai.chat.completions.create.call_count == 1
        sleep.assert_not_called()
        anthropic.messages.create.assert_called_once()

        _close(solwyn)

    def test_retry_after_just_fits_margin_retries(self) -> None:
        # remaining=5.0; Retry-After 4 + 1.0 = 5.0 <= 5.0 -- the retry fits, so the
        # same provider is re-attempted (the other side of the margin boundary).
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("4"), success]

        solwyn = _make_solwyn(openai, model="gpt-5.5", same_provider_retries=1)

        with (
            patch("solwyn.client.Deadline", _FixedDeadline),
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.call_count == 2
        sleep.assert_called_once_with(4.0)

        _close(solwyn)

    def test_retry_then_post_send_ambiguous_reraises_without_failover(self) -> None:
        # A 429 schedules a retry; the re-attempt returns a 5xx (POST_SEND_AMBIGUOUS,
        # may have landed). The double-spend invariant holds: re-raise the original,
        # NO cross-provider failover, exactly one breaker failure.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("0"), _Status(500)]
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.time.sleep"),
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value.status_code == 500
        assert openai.chat.completions.create.call_count == 2
        anthropic.messages.create.assert_not_called()
        assert openai_cb.failure_count == 1

        _close(solwyn)

    def test_retry_then_fail_fast_reraises_and_frees_probe(self) -> None:
        # A 429 schedules a retry; the re-attempt returns a 400 (FAIL_FAST). It
        # propagates and frees the HALF_OPEN probe slot via the neutral
        # release_probe (FAIL_FAST records no health verdict). No fallback so the
        # recovery-eligible primary is the sole candidate that gets the probe (a
        # healthy fallback would otherwise be routed ahead of an OPEN breaker).
        openai = _openai_client()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("0"), _Status(400)]

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )
        openai_cb = solwyn._get_circuit_breaker("openai")
        openai_cb.record_failure()  # -> OPEN; recovery_timeout=0 so admit() probes
        assert openai_cb.state == CircuitState.OPEN

        with (
            patch("solwyn.client.time.sleep"),
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value.status_code == 400
        assert openai.chat.completions.create.call_count == 2
        # FAIL_FAST records no health verdict; the probe slot is freed via the
        # neutral release_probe, leaving the breaker HALF_OPEN but probeable.
        assert openai_cb._half_open_probe_active is False

        _close(solwyn)

    def test_idempotency_never_retries_same_provider_without_crossing(self) -> None:
        # With idempotency="never" the candidate list is the primary only; a 429
        # retry re-attempts the same provider and succeeds without ever calling the
        # fallback -- the "cross-provider unsafe" case the knob originally targeted.
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create.side_effect = [_Status429RetryAfter("0"), success]
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            same_provider_retries=1,
            failover_idempotency="never",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.call_count == 2
        anthropic.messages.create.assert_not_called()
        sleep.assert_called_once_with(0.0)

        _close(solwyn)

    def test_http_date_retry_after_honored_through_loop(self) -> None:
        # The dispatch loop honors the HTTP-date Retry-After form end-to-end (not
        # just the parser): it sleeps the computed delay, then retries.
        openai = _openai_client()
        success = _openai_response()
        future = datetime.now(UTC) + timedelta(seconds=3)
        openai.chat.completions.create.side_effect = [
            _Status429RetryAfter(format_datetime(future)),
            success,
        ]

        solwyn = _make_solwyn(openai, model="gpt-5.5", same_provider_retries=1)

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.call_count == 2
        sleep.assert_called_once()
        assert 1.0 <= sleep.call_args.args[0] <= 3.0

        _close(solwyn)

    def test_same_provider_model_swap_gets_its_own_retry_budget(self) -> None:
        # same_provider_retries is per chain entry. A primary model can fail and
        # count the provider breaker once, then a later same-provider model swap
        # that returns 429 + Retry-After still gets its own same-provider retry.
        client = _openai_client()
        success = _openai_response()
        client.chat.completions.create.side_effect = [
            _Status(429, "primary model failed"),
            _Status429RetryAfter("0"),
            success,
        ]

        solwyn = _make_solwyn(
            client,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(client, "gpt-5.4-mini")],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.time.sleep") as sleep,
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert client.chat.completions.create.call_count == 3
        assert [c.kwargs["model"] for c in client.chat.completions.create.call_args_list] == [
            "gpt-5.5",
            "gpt-5.4-mini",
            "gpt-5.4-mini",
        ]
        sleep.assert_called_once_with(0.0)
        assert openai_cb.failure_count == 0
        assert openai_cb.state == CircuitState.CLOSED

        _close(solwyn)


# ── async (mirror) ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestAsyncSameProviderRetryOn429:
    @pytest.mark.asyncio
    async def test_async_retry_same_provider_then_succeeds(self) -> None:
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create = AsyncMock(side_effect=[_Status429RetryAfter("2"), success])
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        solwyn._reporter.report = MagicMock()
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.asyncio.sleep", new=AsyncMock()) as sleep,
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
        ):
            result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.await_count == 2
        anthropic.messages.create.assert_not_awaited()
        sleep.assert_awaited_once_with(2.0)
        assert openai_cb.failure_count == 0
        assert openai_cb.state == CircuitState.CLOSED

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_retry_exhausted_falls_over(self) -> None:
        openai = _openai_client()
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create = AsyncMock(return_value=anthropic_resp)

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        solwyn._reporter.report = MagicMock()
        openai_cb = solwyn._get_circuit_breaker("openai")
        anthropic_cb = solwyn._get_circuit_breaker("anthropic")
        attempts = 0

        with (
            patch.object(
                openai_cb, "record_failure", wraps=openai_cb.record_failure
            ) as record_failure,
            patch("solwyn.client.asyncio.sleep", new=AsyncMock()) as sleep,
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
        ):

            async def openai_create(*_args: object, **_kwargs: object) -> object:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise _Status429RetryAfter("0")
                assert record_failure.call_count == 0
                raise _Status429RetryAfter("0")

            openai.chat.completions.create = AsyncMock(side_effect=openai_create)
            result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert attempts == 2
        assert openai.chat.completions.create.await_count == 2
        anthropic.messages.create.assert_awaited_once()
        assert result is not anthropic_resp
        assert result.choices[0].message.content == "ok from claude"
        sleep.assert_awaited_once_with(0.0)
        record_failure.assert_called_once_with()
        assert openai_cb.failure_count == 1
        assert anthropic_cb.failure_count == 0

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_retry_after_exceeds_deadline_skips_retry(self) -> None:
        # Async mirror of the deadline-bound guard: over-deadline Retry-After does
        # not sleep or retry; it records the terminal primary failure and falls over.
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status429RetryAfter("30"))
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            same_provider_retries=1,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        solwyn._reporter.report = MagicMock()
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.Deadline", _FixedDeadline),
            patch("solwyn.client.asyncio.sleep", new=AsyncMock()) as sleep,
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
        ):
            result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is not None
        assert openai.chat.completions.create.await_count == 1
        anthropic.messages.create.assert_awaited_once()
        sleep.assert_not_awaited()
        assert openai_cb.failure_count == 1

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_default_zero_fails_over_immediately(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status429RetryAfter("2"))
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        solwyn._reporter.report = MagicMock()
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.asyncio.sleep", new=AsyncMock()) as sleep,
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
        ):
            await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert openai.chat.completions.create.await_count == 1
        anthropic.messages.create.assert_awaited_once()
        sleep.assert_not_awaited()
        assert openai_cb.failure_count == 1

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_half_open_probe_not_stranded_on_retry(self) -> None:
        # Async mirror of the sync HALF_OPEN test: the probe held across the awaited
        # sleep closes the breaker on the retry's success and frees the slot.
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create = AsyncMock(side_effect=[_Status429RetryAfter("0"), success])

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            same_provider_retries=1,
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0,
            circuit_breaker_success_threshold=1,
        )
        solwyn._reporter.report = MagicMock()
        openai_cb = solwyn._get_circuit_breaker("openai")
        openai_cb.record_failure()  # -> OPEN; recovery_timeout=0 so admit() probes
        assert openai_cb.state == CircuitState.OPEN

        with (
            patch("solwyn.client.asyncio.sleep", new=AsyncMock()) as sleep,
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
        ):
            result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert openai.chat.completions.create.await_count == 2
        sleep.assert_awaited_once_with(0.0)
        assert openai_cb.state == CircuitState.CLOSED
        assert openai_cb._half_open_probe_active is False

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()

    @pytest.mark.asyncio
    async def test_async_intermediate_retry_emits_no_error_event(self) -> None:
        # Async mirror: a retried (intermediate) 429 emits NO error event.
        openai = _openai_client()
        success = _openai_response()
        openai.chat.completions.create = AsyncMock(side_effect=[_Status429RetryAfter("0"), success])

        solwyn = AsyncSolwyn(
            openai, api_key=VALID_API_KEY, model="gpt-5.5", same_provider_retries=1
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)
        openai_cb = solwyn._get_circuit_breaker("openai")

        with (
            patch("solwyn.client.asyncio.sleep", new=AsyncMock()),
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
            ),
        ):
            result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert [e for e in events if e.status.value == "error"] == []
        assert len([e for e in events if e.status.value == "success"]) == 1
        assert openai_cb.failure_count == 0

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()
