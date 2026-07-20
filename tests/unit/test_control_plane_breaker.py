"""Shared control-plane circuit breaker: discover a Solwyn outage once.

One ``CircuitBreaker(name="control-plane")`` per client guards BOTH the budget
enforcer's ``/budgets/check`` POST and the reporter's ``/budgets/confirm`` POST.
A streak of failures against Solwyn's own API opens the breaker so the next
check/confirm short-circuits (applies the configured posture / drops the send)
without paying the timeout. Failures on EITHER path count toward the same
breaker; a read-only-key response means Solwyn RESPONDED, so it records success.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY

import solwyn.circuit_breaker as circuit_breaker_mod
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, CircuitState, ProviderName
from solwyn.budget import BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import MetadataReporter


def _breaker(*, failure_threshold: int = 2, recovery_timeout: float = 30.0) -> CircuitBreaker:
    """A control-plane breaker (success_threshold=1, matching production)."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=1,
        name="control-plane",
    )


class _CountingTransport:
    """A switchable httpx transport that counts requests.

    ``mode`` selects the response: ``"connect"`` raises ConnectError (an
    outage), ``"ok"`` returns a valid allow response, ``"read_only"`` returns a
    read-only-key 403 (Solwyn RESPONDED — not an outage).
    """

    def __init__(self, mode: str = "connect") -> None:
        self.mode = mode
        self.count = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.count += 1
        if self.mode == "connect":
            raise httpx.ConnectError("control plane unreachable")
        if self.mode == "read_only":
            return httpx.Response(
                403,
                json={
                    "detail": {
                        "code": "read_only_key",
                        "message": "This API key is read-only and cannot write project data",
                    }
                },
            )
        return httpx.Response(200, json=ALLOW_BUDGET_RESPONSE)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), timeout=5.0)


def _make_enforcer(
    breaker: CircuitBreaker | None,
    transport: _CountingTransport,
    *,
    fail_open: bool = True,
) -> BudgetEnforcer:
    enforcer = BudgetEnforcer(
        "http://control-plane.test",
        VALID_API_KEY,
        fail_open=fail_open,
        control_plane_breaker=breaker,
    )
    enforcer._http = transport.client()
    return enforcer


def _check(enforcer: BudgetEnforcer):
    return enforcer.check_budget(estimated_input_tokens=10, model="gpt-5.5", provider="openai")


def _confirm() -> BudgetConfirmRequest:
    return BudgetConfirmRequest(
        reservation_id="res-cp",
        model="gpt-5.5",
        provider=ProviderName.OPENAI,
        call_id="call-cp",
        token_details=TokenDetails(input_tokens=10, output_tokens=5),
    )


@pytest.mark.unit
class TestCheckBreaker:
    """The enforcer's /budgets/check POST rides the control-plane breaker."""

    def test_open_after_threshold_short_circuits_check_fail_open(self) -> None:
        breaker = _breaker(failure_threshold=2)
        transport = _CountingTransport("connect")
        enforcer = _make_enforcer(breaker, transport, fail_open=True)

        # Two failing checks trip the breaker (each pays one HTTP attempt).
        for _ in range(2):
            result = _check(enforcer)
            assert result.allowed is True  # fail-open posture
        assert transport.count == 2
        assert breaker.get_state().state is CircuitState.OPEN

        # The next check short-circuits: no HTTP attempt, same fail-open posture.
        result = _check(enforcer)
        assert result.allowed is True
        assert transport.count == 2  # flat — the network call was skipped
        enforcer.close()

    def test_open_breaker_applies_local_enforcement_when_fail_closed(self) -> None:
        breaker = _breaker(failure_threshold=2)
        transport = _CountingTransport("connect")
        # fail_open=False + no prior known limit -> local enforcement DENIES.
        enforcer = _make_enforcer(breaker, transport, fail_open=False)

        for _ in range(2):
            assert _check(enforcer).allowed is False
        assert breaker.get_state().state is CircuitState.OPEN

        result = _check(enforcer)
        assert result.allowed is False  # local-enforcement posture, not a network call
        assert transport.count == 2
        enforcer.close()

    def test_recovery_probe_reopens_network_and_success_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"t": 1000.0}
        monkeypatch.setattr(circuit_breaker_mod.time, "monotonic", lambda: clock["t"])

        breaker = _breaker(failure_threshold=2, recovery_timeout=30.0)
        transport = _CountingTransport("connect")
        enforcer = _make_enforcer(breaker, transport, fail_open=True)

        for _ in range(2):
            _check(enforcer)
        assert breaker.get_state().state is CircuitState.OPEN
        assert transport.count == 2

        # Before the recovery window elapses, the check stays short-circuited.
        clock["t"] += 10.0
        _check(enforcer)
        assert transport.count == 2

        # After the window elapses, the next check probes the network again...
        clock["t"] += 25.0
        transport.mode = "ok"
        result = _check(enforcer)
        assert transport.count == 3  # HTTP was attempted (half-open probe)
        assert result.allowed is True
        # ...and a successful probe closes the breaker (success_threshold=1).
        assert breaker.get_state().state is CircuitState.CLOSED
        enforcer.close()

    def test_read_only_key_records_success_and_keeps_breaker_closed(self) -> None:
        breaker = _breaker(failure_threshold=2)
        transport = _CountingTransport("read_only")
        enforcer = _make_enforcer(breaker, transport, fail_open=True)

        # A read-only-key 403 means the control plane RESPONDED: it fails open
        # but never counts as an outage, so the breaker stays closed.
        for _ in range(3):
            assert _check(enforcer).allowed is True
        assert transport.count == 3
        state = breaker.get_state()
        assert state.state is CircuitState.CLOSED
        assert state.failure_count == 0
        enforcer.close()

    def test_no_breaker_never_short_circuits(self) -> None:
        transport = _CountingTransport("connect")
        enforcer = _make_enforcer(None, transport, fail_open=True)

        # With no breaker every check hits the network (today's behavior).
        for _ in range(4):
            assert _check(enforcer).allowed is True
        assert transport.count == 4
        enforcer.close()


def _make_reporter(
    breaker: CircuitBreaker | None, transport: _CountingTransport
) -> MetadataReporter:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(
            "http://control-plane.test",
            VALID_API_KEY,
            control_plane_breaker=breaker,
        )
    reporter._shutdown.set()
    reporter._thread.join(timeout=2.0)
    reporter._http = transport.client()
    return reporter


@pytest.mark.unit
class TestConfirmBreaker:
    """The reporter's /budgets/confirm POST rides the SAME control-plane breaker."""

    def test_confirm_failures_open_shared_breaker_then_skip_send(self) -> None:
        breaker = _breaker(failure_threshold=2)
        transport = _CountingTransport("connect")
        reporter = _make_reporter(breaker, transport)

        for _ in range(2):
            reporter._send_confirm(_confirm())
        assert transport.count == 2
        assert breaker.get_state().state is CircuitState.OPEN
        assert reporter._consecutive_confirm_failures == 2

        # A subsequent confirm makes NO HTTP attempt (breaker open)...
        reporter._send_confirm(_confirm())
        assert transport.count == 2
        # ...and the skip does not advance the confirm-failure counter.
        assert reporter._consecutive_confirm_failures == 2
        reporter._http.close()

    def test_check_and_confirm_failures_share_one_breaker(self) -> None:
        # One failure on the CHECK path + one on the CONFIRM path opens a
        # threshold-2 breaker: they are the same health domain.
        breaker = _breaker(failure_threshold=2)
        check_transport = _CountingTransport("connect")
        confirm_transport = _CountingTransport("connect")
        enforcer = _make_enforcer(breaker, check_transport, fail_open=True)
        reporter = _make_reporter(breaker, confirm_transport)

        _check(enforcer)
        assert breaker.get_state().state is CircuitState.CLOSED
        reporter._send_confirm(_confirm())
        assert breaker.get_state().state is CircuitState.OPEN

        # Both paths now short-circuit.
        _check(enforcer)
        reporter._send_confirm(_confirm())
        assert check_transport.count == 1
        assert confirm_transport.count == 1
        enforcer.close()
        reporter._http.close()

    def test_read_only_key_on_confirm_records_success(self) -> None:
        breaker = _breaker(failure_threshold=2)
        transport = _CountingTransport("read_only")
        reporter = _make_reporter(breaker, transport)

        for _ in range(3):
            reporter._send_confirm(_confirm())
        assert transport.count == 3
        assert breaker.get_state().state is CircuitState.CLOSED
        assert reporter._consecutive_confirm_failures == 0
        reporter._http.close()

    def test_no_breaker_confirm_never_short_circuits(self) -> None:
        transport = _CountingTransport("connect")
        reporter = _make_reporter(None, transport)

        for _ in range(3):
            reporter._send_confirm(_confirm())
        assert transport.count == 3
        reporter._http.close()
