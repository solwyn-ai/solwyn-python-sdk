"""Control-plane brownout: discover the outage once, never withhold a response.

Against a black-holed Solwyn API (TCP connects land in the listen backlog and
nothing ever answers), a non-streaming call must cost the caller at most the
~1s budget pre-flight timeout ONCE; the shared control-plane breaker then
applies the configured posture instantly for the cool-down window. Settlement
must never gate the provider response — it rides the reporter queue.

These are real-socket timing tests: bounds are generous (CI slack) but far
below the pre-fix behavior (5s check + 5s blocking confirm, on EVERY call).
"""

from __future__ import annotations

import socket
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._types import CircuitState
from solwyn.client import AsyncSolwyn, Solwyn

# Upper bound for a call that pays the budget pre-flight timeout exactly once
# (1s default) plus CI slack. Pre-fix this path cost ~5s (check) and, with a
# reservation, another ~5s (blocking confirm).
FIRST_CALL_BOUND = 2.5
# Upper bound for a call inside the breaker cool-down window: no Solwyn I/O.
SHORT_CIRCUIT_BOUND = 0.6
# Upper bound for a call whose ONLY Solwyn exposure would be settlement: the
# confirm must be queued, never awaited.
CONFIRM_NEVER_GATES_BOUND = 1.5


@pytest.fixture
def black_hole_url():
    """A real listening socket that never accepts or answers."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.close()


def _openai_client(*, is_async: bool = False) -> tuple[MagicMock, SimpleNamespace]:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "AsyncOpenAI" if is_async else "OpenAI"
    client.with_options.return_value = client
    message = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    response = SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    if is_async:
        client.chat.completions.create = AsyncMock(return_value=response)
    else:
        client.chat.completions.create.return_value = response
    return client, response


def _allow_budget(reservation_id: str | None = "res_123") -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=reservation_id,
        project_id=None,
        budget_limit=100.0,
        current_usage=0.0,
        mode=SimpleNamespace(value="alert_only"),
        price_hints=None,
    )


def _teardown_sync(solwyn: Solwyn) -> None:
    """Drop queued I/O (it would flush into the black hole) and shut down."""
    reporter = solwyn._solwyn_reporter
    reporter._queue.clear()
    reporter._confirm_queue.clear()
    reporter._settlement_queue.clear()
    reporter._shutdown.set()
    reporter._thread.join(timeout=2.0)
    reporter._http.close()
    solwyn._solwyn_budget._http.close()


async def _teardown_async(solwyn: AsyncSolwyn) -> None:
    reporter = solwyn._solwyn_reporter
    reporter._queue.clear()
    reporter._confirm_queue.clear()
    reporter._settlement_queue.clear()
    await reporter.close()
    await solwyn._solwyn_budget._http.aclose()


_PLAIN_REQUEST = {
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.mark.unit
class TestSyncBrownout:
    def test_outage_discovered_once_then_short_circuits(self, black_hole_url) -> None:
        client, response = _openai_client()
        solwyn = Solwyn(
            client,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            api_url=black_hole_url,
            control_plane_failure_threshold=1,
            reporter_flush_interval=3600.0,
        )
        try:
            start = time.monotonic()
            first = solwyn.chat.completions.create(**_PLAIN_REQUEST)
            first_elapsed = time.monotonic() - start

            start = time.monotonic()
            second = solwyn.chat.completions.create(**_PLAIN_REQUEST)
            second_elapsed = time.monotonic() - start

            # The paid provider response is returned both times.
            assert first is response
            assert second is response
            # First call pays the (short) pre-flight timeout once...
            assert first_elapsed < FIRST_CALL_BOUND, (
                f"first brownout call took {first_elapsed:.2f}s — the budget "
                f"pre-flight must be bounded by ~1s and confirm must not block"
            )
            # ...which opens the shared control-plane breaker...
            assert solwyn._solwyn_control_plane_breaker.get_state().state is CircuitState.OPEN
            # ...so the next call applies the posture instantly.
            assert second_elapsed < SHORT_CIRCUIT_BOUND, (
                f"second brownout call took {second_elapsed:.2f}s — the breaker "
                f"must short-circuit outage rediscovery inside the cool-down"
            )
        finally:
            _teardown_sync(solwyn)

    def test_confirm_never_gates_response(self, black_hole_url) -> None:
        client, response = _openai_client()
        solwyn = Solwyn(
            client,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            api_url=black_hole_url,
            reporter_flush_interval=3600.0,
        )
        try:
            with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
                start = time.monotonic()
                result = solwyn.chat.completions.create(**_PLAIN_REQUEST)
                elapsed = time.monotonic() - start

            assert result is response
            assert elapsed < CONFIRM_NEVER_GATES_BOUND, (
                f"call with a reservation took {elapsed:.2f}s against a "
                f"black-holed API — confirm must be queued, never awaited"
            )
            # The settlement (confirm + event, one ordered item) was queued.
            assert len(solwyn._solwyn_reporter._settlement_queue) == 1
            settlement = solwyn._solwyn_reporter._settlement_queue[0]
            confirm, event = settlement.confirm.request, settlement.event
            assert confirm.reservation_id == "res_123"
            assert confirm.call_id == event.call_id
        finally:
            _teardown_sync(solwyn)


@pytest.mark.unit
class TestAsyncBrownout:
    @pytest.mark.asyncio
    async def test_outage_discovered_once_then_short_circuits(self, black_hole_url) -> None:
        client, response = _openai_client(is_async=True)
        solwyn = AsyncSolwyn(
            client,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            api_url=black_hole_url,
            control_plane_failure_threshold=1,
            reporter_flush_interval=3600.0,
        )
        try:
            start = time.monotonic()
            first = await solwyn.chat.completions.create(**_PLAIN_REQUEST)
            first_elapsed = time.monotonic() - start

            start = time.monotonic()
            second = await solwyn.chat.completions.create(**_PLAIN_REQUEST)
            second_elapsed = time.monotonic() - start

            assert first is response
            assert second is response
            assert first_elapsed < FIRST_CALL_BOUND
            assert solwyn._solwyn_control_plane_breaker.get_state().state is CircuitState.OPEN
            assert second_elapsed < SHORT_CIRCUIT_BOUND
        finally:
            await _teardown_async(solwyn)

    @pytest.mark.asyncio
    async def test_confirm_never_gates_response(self, black_hole_url) -> None:
        client, response = _openai_client(is_async=True)
        solwyn = AsyncSolwyn(
            client,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            api_url=black_hole_url,
            reporter_flush_interval=3600.0,
        )
        try:
            with patch.object(
                solwyn._solwyn_budget, "check_budget", AsyncMock(return_value=_allow_budget())
            ):
                start = time.monotonic()
                result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)
                elapsed = time.monotonic() - start

            assert result is response
            assert elapsed < CONFIRM_NEVER_GATES_BOUND
            assert len(solwyn._solwyn_reporter._settlement_queue) == 1
            settlement = solwyn._solwyn_reporter._settlement_queue[0]
            confirm, event = settlement.confirm.request, settlement.event
            assert confirm.reservation_id == "res_123"
            assert confirm.call_id == event.call_id
        finally:
            await _teardown_async(solwyn)
