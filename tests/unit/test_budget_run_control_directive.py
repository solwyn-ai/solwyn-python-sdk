"""Server-pushed run-control behavior across check and lease channels."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID, call_uuid

import solwyn.budget as budget_module
from solwyn import _run_control
from solwyn._lease import GrantOutcome
from solwyn._run_control import mark_terminated, run_termination
from solwyn._types import BudgetCheckResponse, BudgetMode, LeaseGrantResponse
from solwyn.budget import (
    AsyncBudgetEnforcer,
    BudgetCheckResult,
    BudgetEnforcer,
    _BudgetEnforcerBase,
)

RUN_ID = "run-directed"
OTHER_RUN_ID = "run-other"


@pytest.fixture(autouse=True)
def _clean_run_control_registry() -> Iterator[None]:
    with _run_control._STATE.lock:
        _run_control._STATE._clear_for_test_locked()
    yield
    with _run_control._STATE.lock:
        _run_control._STATE._clear_for_test_locked()


def _check_payload(
    *,
    allowed: bool,
    run_id: str = RUN_ID,
    reason: str = "manual_kill",
) -> dict[str, object]:
    return {
        **ALLOW_BUDGET_RESPONSE,
        "allowed": allowed,
        "reservation_id": "res_123" if allowed else None,
        "mode": "alert_only" if allowed else "hard_deny",
        "denied_by_period": None if allowed else "run_stopped",
        "run_control": {
            "version": "1",
            "action": "terminate",
            "agent_run_id": run_id,
            "reason": reason,
        },
    }


def _response(payload: dict[str, object]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


def _enforcer() -> BudgetEnforcer:
    return BudgetEnforcer(
        api_url="https://api.test.solwyn.ai",
        api_key=VALID_API_KEY,
        lease_enabled=False,
    )


def _async_enforcer() -> AsyncBudgetEnforcer:
    return AsyncBudgetEnforcer(
        api_url="https://api.test.solwyn.ai",
        api_key=VALID_API_KEY,
        lease_enabled=False,
    )


def _grant_payload(
    *,
    lease_id: str = "lse_a",
    generation: int = 1,
    directive_run_id: str | None = None,
    directive_reason: str = "manual_kill",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "eligible": True,
        "allowed": True,
        "lease_id": lease_id,
        "generation": generation,
        "granted_tokens": 32_000,
        "refresh_interval_s": 15.0,
        "lease_length_s": 120.0,
        "headroom_share_tokens": 500_000,
        "posture": {"mode": "hard_deny", "on_unreachable": "fail_open"},
        "final_grant": False,
        "project_id": VALID_PROJECT_ID,
        "mode": "hard_deny",
        "budget_limit": 100.0,
        "current_usage": 2.0,
        "remaining_budget": 98.0,
    }
    if directive_run_id is not None:
        payload["run_control"] = {
            "version": "1",
            "action": "terminate",
            "agent_run_id": directive_run_id,
            "reason": directive_reason,
        }
    return payload


@pytest.mark.unit
class TestCheckRunControl:
    def test_matching_denied_directive_marks_registry_and_sticks(self) -> None:
        enforcer = _enforcer()

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(_check_payload(allowed=False)),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.reason, termination.source) == ("manual_kill", "server")
        actual = (result.allowed, result.deny_source, result.deny_reason, result.denied_by_period)
        assert actual == (
            False,
            "server",
            "manual_kill",
            "run_stopped",
        )
        replay = enforcer._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert (replay.deny_source, replay.deny_reason) == ("sticky_replay", "manual_kill")

    def test_repeated_sync_directive_keeps_first_server_reason_everywhere(self) -> None:
        enforcer = _enforcer()
        responses = [
            _response(_check_payload(allowed=False, reason="first_reason")),
            _response(_check_payload(allowed=False, reason="second_reason")),
        ]

        with patch.object(enforcer._http, "post", side_effect=responses):
            first = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
            second = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.source, termination.reason) == ("server", "first_reason")
        assert (first.deny_source, first.deny_reason) == ("server", "first_reason")
        assert (second.deny_source, second.deny_reason) == ("server", "first_reason")
        sticky = enforcer._run_hard_deny_responses[RUN_ID]
        assert sticky.run_control is not None
        assert sticky.run_control.reason == "first_reason"

        for index in range(256):
            mark_terminated(
                f"first-writer-registry-churn-{index}",
                reason="other_reason",
                source="server",
            )
        assert run_termination(RUN_ID) is None
        replay = enforcer._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert (replay.deny_source, replay.deny_reason) == (
            "sticky_replay",
            "first_reason",
        )

    def test_sync_outage_fail_opens_after_both_exact_lrus_evict_server_stop(self) -> None:
        enforcer = _enforcer()
        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(_check_payload(allowed=False)),
        ):
            first = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        assert first.allowed is False

        for index in range(257):
            churn_run = f"sync-outage-churn-{index}"
            enforcer._apply_check_run_control(
                BudgetCheckResponse.model_validate(_check_payload(allowed=False, run_id=churn_run)),
                churn_run,
            )
        assert RUN_ID not in _run_control._STATE.terminations
        assert RUN_ID not in enforcer._run_hard_deny_responses

        with patch.object(
            enforcer._http,
            "post",
            side_effect=httpx.ConnectError("control plane down"),
        ):
            outage = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        assert outage.allowed is True
        assert outage.denied_by_period is None

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(ALLOW_BUDGET_RESPONSE),
        ):
            recovered = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        assert recovered.allowed is True
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

        with patch.object(
            enforcer._http,
            "post",
            side_effect=httpx.ConnectError("control plane down again"),
        ):
            after_ordered_allow = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        assert after_ordered_allow.allowed is True

    @pytest.mark.asyncio
    async def test_repeated_async_directive_keeps_first_server_reason(self) -> None:
        enforcer = _async_enforcer()
        enforcer._http.post = AsyncMock(
            side_effect=[
                _response(_check_payload(allowed=False, reason="first_reason")),
                _response(_check_payload(allowed=False, reason="second_reason")),
            ]
        )

        first = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )
        second = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )

        assert (first.deny_source, first.deny_reason) == ("server", "first_reason")
        assert (second.deny_source, second.deny_reason) == ("server", "first_reason")
        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.source, termination.reason) == ("server", "first_reason")
        await enforcer._http.aclose()

    @pytest.mark.asyncio
    async def test_async_outage_fail_opens_after_both_exact_lrus_evict_server_stop(self) -> None:
        enforcer = _async_enforcer()
        enforcer._http.post = AsyncMock(return_value=_response(_check_payload(allowed=False)))
        first = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )
        assert first.allowed is False

        for index in range(257):
            churn_run = f"async-outage-churn-{index}"
            enforcer._apply_check_run_control(
                BudgetCheckResponse.model_validate(_check_payload(allowed=False, run_id=churn_run)),
                churn_run,
            )
        assert RUN_ID not in _run_control._STATE.terminations
        assert RUN_ID not in enforcer._run_hard_deny_responses

        enforcer._http.post = AsyncMock(side_effect=httpx.ConnectError("control plane down"))
        outage = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )

        assert outage.allowed is True
        assert outage.denied_by_period is None

        enforcer._http.post = AsyncMock(return_value=_response(ALLOW_BUDGET_RESPONSE))
        recovered = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )
        assert recovered.allowed is True
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

        enforcer._http.post = AsyncMock(side_effect=httpx.ConnectError("control plane down again"))
        after_ordered_allow = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )
        assert after_ordered_allow.allowed is True
        await enforcer._http.aclose()

    def test_delayed_sync_allow_cannot_clear_a_newer_directive(self) -> None:
        enforcer = _enforcer()
        allow_started = threading.Event()
        release_allow = threading.Event()
        response_lock = threading.Lock()
        response_index = 0
        first_result: list[BudgetCheckResult] = []

        def post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal response_index
            with response_lock:
                current = response_index
                response_index += 1
            if current == 0:
                allow_started.set()
                assert release_allow.wait(timeout=2)
                return _response(dict(ALLOW_BUDGET_RESPONSE))
            return _response(_check_payload(allowed=False))

        def run_delayed_allow() -> None:
            first_result.append(
                enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id=RUN_ID,
                )
            )

        with patch.object(enforcer._http, "post", side_effect=post):
            first = threading.Thread(target=run_delayed_allow)
            first.start()
            assert allow_started.wait(timeout=2)
            stopped = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
            release_allow.set()
            first.join(timeout=2)

        assert not first.is_alive()
        assert stopped.allowed is False
        assert len(first_result) == 1
        delayed = first_result[0]
        assert delayed.allowed is False
        assert delayed.deny_reason == "manual_kill"
        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.source, termination.reason) == ("server", "manual_kill")
        replay = enforcer._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert (replay.denied_by_period, replay.deny_reason) == (
            "run_stopped",
            "manual_kill",
        )

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(dict(ALLOW_BUDGET_RESPONSE)),
        ):
            later = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        assert later.allowed is True
        assert run_termination(RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

    @pytest.mark.asyncio
    async def test_delayed_async_allow_cannot_clear_a_newer_directive(self) -> None:
        enforcer = _async_enforcer()
        allow_started = asyncio.Event()
        release_allow = asyncio.Event()
        response_index = 0

        async def post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal response_index
            current = response_index
            response_index += 1
            if current == 0:
                allow_started.set()
                await asyncio.wait_for(release_allow.wait(), timeout=2)
                return _response(dict(ALLOW_BUDGET_RESPONSE))
            return _response(_check_payload(allowed=False))

        enforcer._http.post = AsyncMock(side_effect=post)
        delayed_task = asyncio.create_task(
            enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        )
        await asyncio.wait_for(allow_started.wait(), timeout=2)
        stopped = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )
        release_allow.set()
        delayed = await asyncio.wait_for(delayed_task, timeout=2)

        assert stopped.allowed is False
        assert (delayed.allowed, delayed.deny_reason) == (False, "manual_kill")
        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.source, termination.reason) == ("server", "manual_kill")
        replay = enforcer._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert replay.deny_reason == "manual_kill"

        enforcer._http.post = AsyncMock(return_value=_response(dict(ALLOW_BUDGET_RESPONSE)))
        later = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )
        assert later.allowed is True
        assert run_termination(RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None
        await enforcer._http.aclose()

    def test_delayed_allow_on_another_enforcer_observes_newer_registry_stop(self) -> None:
        delayed_enforcer = _enforcer()
        stopping_enforcer = _enforcer()
        # A prior server stop is old enough for A to clear. B's repeated
        # directive must publish a newer observation without changing the
        # first-writer record's public reason/source.
        mark_terminated(RUN_ID, reason="manual_kill", source="server")
        allow_started = threading.Event()
        release_allow = threading.Event()
        delayed_result: list[BudgetCheckResult] = []

        def delayed_post(*args: object, **kwargs: object) -> MagicMock:
            allow_started.set()
            assert release_allow.wait(timeout=2)
            return _response(dict(ALLOW_BUDGET_RESPONSE))

        def run_delayed_allow() -> None:
            delayed_result.append(
                delayed_enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id=RUN_ID,
                )
            )

        with (
            patch.object(delayed_enforcer._http, "post", side_effect=delayed_post),
            patch.object(
                stopping_enforcer._http,
                "post",
                return_value=_response(_check_payload(allowed=False)),
            ),
        ):
            first = threading.Thread(target=run_delayed_allow)
            first.start()
            assert allow_started.wait(timeout=2)
            stopped = stopping_enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
            release_allow.set()
            first.join(timeout=2)

        assert stopped.allowed is False
        assert not first.is_alive()
        assert len(delayed_result) == 1
        assert delayed_result[0].allowed is False
        assert delayed_result[0].deny_reason == "manual_kill"
        assert run_termination(RUN_ID) is not None
        replay = delayed_enforcer._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert replay.deny_reason == "manual_kill"

    def test_equal_epoch_same_enforcer_sticky_stop_wins_then_later_allow_clears(
        self,
    ) -> None:
        enforcer = _enforcer()
        allow_started = threading.Event()
        release_allow = threading.Event()
        response_index = 0
        delayed_result: list[BudgetCheckResult] = []

        def post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal response_index
            current = response_index
            response_index += 1
            if current == 0:
                allow_started.set()
                assert release_allow.wait(timeout=2)
                return _response(dict(ALLOW_BUDGET_RESPONSE))
            return _response(_check_payload(allowed=False))

        def run_delayed_allow() -> None:
            delayed_result.append(
                enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id=RUN_ID,
                )
            )

        with (
            patch.object(budget_module.time, "monotonic", return_value=100.0),
            patch.object(enforcer._http, "post", side_effect=post),
        ):
            first = threading.Thread(target=run_delayed_allow)
            first.start()
            assert allow_started.wait(timeout=2)
            stopped = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
            # Isolate the same-enforcer sticky epoch: the registry record is
            # deliberately evicted while both observations remain exactly 100.
            for index in range(256):
                mark_terminated(
                    f"equal-sticky-churn-{index}",
                    reason="manual_kill",
                    source="server",
                )
            assert run_termination(RUN_ID) is None
            release_allow.set()
            first.join(timeout=2)

        assert stopped.allowed is False
        assert not first.is_alive()
        assert len(delayed_result) == 1
        assert (delayed_result[0].allowed, delayed_result[0].deny_reason) == (
            False,
            "manual_kill",
        )

        with (
            patch.object(budget_module.time, "monotonic", return_value=101.0),
            patch.object(
                enforcer._http,
                "post",
                return_value=_response(dict(ALLOW_BUDGET_RESPONSE)),
            ),
        ):
            later = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        assert later.allowed is True
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

    def test_equal_epoch_cross_enforcer_registry_stop_wins_then_later_allow_clears(
        self,
    ) -> None:
        delayed_enforcer = _enforcer()
        stopping_enforcer = _enforcer()
        allow_started = threading.Event()
        release_allow = threading.Event()
        delayed_result: list[BudgetCheckResult] = []

        def delayed_post(*args: object, **kwargs: object) -> MagicMock:
            allow_started.set()
            assert release_allow.wait(timeout=2)
            return _response(dict(ALLOW_BUDGET_RESPONSE))

        def run_delayed_allow() -> None:
            delayed_result.append(
                delayed_enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id=RUN_ID,
                )
            )

        with (
            patch.object(budget_module.time, "monotonic", return_value=100.0),
            patch.object(delayed_enforcer._http, "post", side_effect=delayed_post),
            patch.object(
                stopping_enforcer._http,
                "post",
                return_value=_response(_check_payload(allowed=False)),
            ),
        ):
            first = threading.Thread(target=run_delayed_allow)
            first.start()
            assert allow_started.wait(timeout=2)
            stopped = stopping_enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
            release_allow.set()
            first.join(timeout=2)

        assert stopped.allowed is False
        assert not first.is_alive()
        assert len(delayed_result) == 1
        assert (delayed_result[0].allowed, delayed_result[0].deny_reason) == (
            False,
            "manual_kill",
        )
        assert run_termination(RUN_ID) is not None

        with (
            patch.object(budget_module.time, "monotonic", return_value=101.0),
            patch.object(
                delayed_enforcer._http,
                "post",
                return_value=_response(dict(ALLOW_BUDGET_RESPONSE)),
            ),
        ):
            later = delayed_enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        assert later.allowed is True
        assert run_termination(RUN_ID) is None
        assert delayed_enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

    def test_delayed_allow_cannot_clear_a_newer_run_cap_denial(self) -> None:
        enforcer = _enforcer()
        allow_started = threading.Event()
        release_allow = threading.Event()
        response_index = 0
        delayed_result: list[BudgetCheckResult] = []
        run_deny = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "reservation_id": None,
            "mode": "hard_deny",
            "denied_by_period": "agent_run",
        }

        def post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal response_index
            current = response_index
            response_index += 1
            if current == 0:
                allow_started.set()
                assert release_allow.wait(timeout=2)
                return _response(dict(ALLOW_BUDGET_RESPONSE))
            return _response(run_deny)

        def run_delayed_allow() -> None:
            delayed_result.append(
                enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id=RUN_ID,
                )
            )

        with patch.object(enforcer._http, "post", side_effect=post):
            first = threading.Thread(target=run_delayed_allow)
            first.start()
            assert allow_started.wait(timeout=2)
            denied = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
            release_allow.set()
            first.join(timeout=2)

        assert denied.allowed is False
        assert not first.is_alive()
        assert len(delayed_result) == 1
        delayed = delayed_result[0]
        assert delayed.allowed is False
        assert delayed.denied_by_period == "agent_run"
        assert delayed.deny_reason == "agent_run"

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(dict(ALLOW_BUDGET_RESPONSE)),
        ):
            later = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        assert later.allowed is True
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

    def test_sticky_replay_retains_reason_after_registry_lru_eviction(self) -> None:
        enforcer = _enforcer()
        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(_check_payload(allowed=False)),
        ):
            enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        for index in range(256):
            mark_terminated(
                f"registry-churn-{index}",
                reason="manual_kill",
                source="server",
            )
        assert run_termination(RUN_ID) is None

        with patch.object(
            enforcer._http,
            "post",
            side_effect=httpx.ConnectError("control plane unavailable"),
        ):
            replay = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        assert (replay.allowed, replay.deny_source, replay.deny_reason) == (
            False,
            "sticky_replay",
            "manual_kill",
        )
        assert replay.denied_by_period == "run_stopped"

    def test_allow_shaped_directive_is_synthesized_as_hard_run_denial(self) -> None:
        enforcer = _enforcer()

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(_check_payload(allowed=True)),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        assert (result.allowed, result.mode, result.denied_by_period, result.deny_reason) == (
            False,
            BudgetMode.HARD_DENY,
            "run_stopped",
            "manual_kill",
        )
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is not None

    def test_mismatched_directive_warns_once_and_has_no_control_effect(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _enforcer()

        with (
            caplog.at_level(logging.WARNING, logger="solwyn.budget"),
            patch.object(
                enforcer._http,
                "post",
                return_value=_response(_check_payload(allowed=True, run_id=OTHER_RUN_ID)),
            ),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        assert result.allowed is True
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        assert enforcer._last_known_budget_limit is None
        records = [r for r in caplog.records if "run_control.directive_run_mismatch" in r.message]
        assert len(records) == 1
        assert RUN_ID in records[0].message
        assert OTHER_RUN_ID in records[0].message

    def test_mismatched_control_denial_uses_outage_posture_not_run_stop(self) -> None:
        enforcer = _enforcer()

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(_check_payload(allowed=False, run_id=OTHER_RUN_ID)),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        assert result.allowed is True
        assert result.denied_by_period is None
        assert result.warning == "Cloud API unreachable; proceeding in fail-open mode"
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

    def test_directive_without_requested_run_warns_and_has_no_control_effect(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _enforcer()

        with (
            caplog.at_level(logging.WARNING, logger="solwyn.budget"),
            patch.object(
                enforcer._http,
                "post",
                return_value=_response(_check_payload(allowed=True)),
            ),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
            )

        assert result.allowed is True
        assert run_termination(RUN_ID) is None
        assert enforcer._last_known_budget_limit is None
        records = [r for r in caplog.records if "run_control.directive_run_mismatch" in r.message]
        assert len(records) == 1
        assert "request_agent_run_id=None" in records[0].message
        assert RUN_ID in records[0].message

    def test_later_live_allow_clears_server_registry_and_sticky_only(self) -> None:
        enforcer = _enforcer()
        responses = [
            _response(_check_payload(allowed=False)),
            _response(dict(ALLOW_BUDGET_RESPONSE)),
        ]

        with patch.object(enforcer._http, "post", side_effect=responses):
            enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
            allowed = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        assert allowed.allowed is True
        assert run_termination(RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

        mark_terminated(RUN_ID, reason="velocity:repeat_size", source="local_velocity")
        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(dict(ALLOW_BUDGET_RESPONSE)),
        ):
            enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )
        assert run_termination(RUN_ID) is not None

    def test_server_directive_cannot_overwrite_a_local_first_writer(self) -> None:
        enforcer = _enforcer()
        mark_terminated(RUN_ID, reason="velocity:repeat_size", source="local_velocity")

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(_check_payload(allowed=False)),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
            )

        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.reason, termination.source) == (
            "velocity:repeat_size",
            "local_velocity",
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_async_matching_directive_applies_on_same_response(self) -> None:
        enforcer = _async_enforcer()
        enforcer._http.post = AsyncMock(return_value=_response(_check_payload(allowed=False)))

        result = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )

        assert result.allowed is False
        assert result.deny_reason == "manual_kill"
        assert run_termination(RUN_ID) is not None
        await enforcer._http.aclose()

    @pytest.mark.asyncio
    async def test_async_mismatched_control_denial_uses_outage_posture(self) -> None:
        enforcer = _async_enforcer()
        enforcer._http.post = AsyncMock(
            return_value=_response(_check_payload(allowed=False, run_id=OTHER_RUN_ID))
        )

        result = await enforcer.check_budget(
            estimated_input_tokens=10,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
        )

        assert result.allowed is True
        assert result.denied_by_period is None
        assert run_termination(RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None
        await enforcer._http.aclose()


@pytest.mark.unit
class TestLeaseRunControl:
    def test_sync_grant_directive_denies_through_full_admission_path(self) -> None:
        enforcer = BudgetEnforcer(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(_grant_payload(directive_run_id=RUN_ID)),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                estimated_output_bound=100,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
                call_id=call_uuid("directed-grant-sync"),
            )

        assert (result.allowed, result.mode, result.denied_by_period, result.deny_reason) == (
            False,
            BudgetMode.HARD_DENY,
            "run_stopped",
            "manual_kill",
        )
        assert enforcer._lease.lease_id_for(RUN_ID) is None

    @pytest.mark.asyncio
    async def test_async_grant_directive_denies_through_full_admission_path(self) -> None:
        enforcer = AsyncBudgetEnforcer(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        enforcer._http.post = AsyncMock(
            return_value=_response(_grant_payload(directive_run_id=RUN_ID))
        )

        result = await enforcer.check_budget(
            estimated_input_tokens=10,
            estimated_output_bound=100,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
            call_id=call_uuid("directed-grant-async"),
        )

        assert (result.allowed, result.mode, result.denied_by_period, result.deny_reason) == (
            False,
            BudgetMode.HARD_DENY,
            "run_stopped",
            "manual_kill",
        )
        assert enforcer._lease.lease_id_for(RUN_ID) is None
        await enforcer._http.aclose()

    @pytest.mark.asyncio
    async def test_async_wrong_run_control_denied_grant_uses_outage_posture(self) -> None:
        enforcer = AsyncBudgetEnforcer(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        payload = _grant_payload(directive_run_id=OTHER_RUN_ID)
        payload.update(
            {
                "allowed": False,
                "denied_by_period": "run_stopped",
                "lease_id": None,
                "generation": None,
                "granted_tokens": None,
                "refresh_interval_s": None,
                "lease_length_s": None,
                "headroom_share_tokens": None,
                "posture": None,
                "final_grant": None,
            }
        )
        enforcer._http.post = AsyncMock(return_value=_response(payload))

        result = await enforcer.check_budget(
            estimated_input_tokens=10,
            estimated_output_bound=100,
            model="gpt-5.5",
            provider="openai",
            agent_run_id=RUN_ID,
            call_id=call_uuid("wrong-run-control-denied-grant-async"),
        )

        assert result.allowed is True
        assert result.denied_by_period is None
        assert enforcer._lease.lease_id_for(RUN_ID) is None
        assert run_termination(RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None
        await enforcer._http.aclose()

    def test_grant_directive_marks_and_refuses_before_install(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        other = LeaseGrantResponse.model_validate(_grant_payload(lease_id="lse_other"))
        assert (
            base._apply_lease_response(OTHER_RUN_ID, other, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        )
        response = LeaseGrantResponse.model_validate(_grant_payload(directive_run_id=RUN_ID))
        response = response.model_copy(update={"mode": BudgetMode.ALERT_ONLY})

        outcome, _ = base._apply_lease_response(RUN_ID, response, ["gpt-5.5"])

        assert outcome is GrantOutcome.DENIED
        assert base._lease.lease_id_for(RUN_ID) is None
        assert base._lease.lease_id_for(OTHER_RUN_ID) == "lse_other"
        assert run_termination(RUN_ID) is not None
        assert base._build_prior_hard_deny_unavailable_result(RUN_ID) is not None
        sticky = base._run_hard_deny_responses[RUN_ID]
        assert sticky.mode is BudgetMode.HARD_DENY
        assert base._lease_path_applies(RUN_ID) is False

    def test_real_stopped_deny_shape_keeps_directive_reason_and_run_scope(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        payload = _grant_payload(directive_run_id=RUN_ID)
        payload.update(
            {
                "allowed": False,
                "denied_by_period": "run_stopped",
                "lease_id": None,
                "generation": None,
                "granted_tokens": None,
                "refresh_interval_s": None,
                "lease_length_s": None,
                "headroom_share_tokens": None,
                "posture": None,
                "final_grant": None,
            }
        )
        response = LeaseGrantResponse.model_validate(payload)

        outcome, _ = base._apply_lease_response(RUN_ID, response, ["gpt-5.5"])
        result = base._lease_deny_result(RUN_ID, response)

        assert outcome is GrantOutcome.DENIED
        assert (result.deny_source, result.deny_reason, result.denied_by_period) == (
            "server",
            "manual_kill",
            "run_stopped",
        )
        assert base._build_prior_hard_deny_unavailable_result(OTHER_RUN_ID) is None

    def test_repeated_grant_directive_keeps_first_server_reason_everywhere(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        first = LeaseGrantResponse.model_validate(
            _grant_payload(
                directive_run_id=RUN_ID,
                directive_reason="first_reason",
            )
        )
        second = LeaseGrantResponse.model_validate(
            _grant_payload(
                directive_run_id=RUN_ID,
                directive_reason="second_reason",
            )
        )

        assert base._apply_lease_response(RUN_ID, first, ["gpt-5.5"])[0] is GrantOutcome.DENIED
        assert base._apply_lease_response(RUN_ID, second, ["gpt-5.5"])[0] is GrantOutcome.DENIED

        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.source, termination.reason) == ("server", "first_reason")
        assert second.run_control is not None
        assert second.run_control.reason == "first_reason"
        same_call = base._lease_deny_result(RUN_ID, second)
        assert (same_call.deny_source, same_call.deny_reason) == (
            "server",
            "first_reason",
        )
        sticky = base._run_hard_deny_responses[RUN_ID]
        assert sticky.run_control is not None
        assert sticky.run_control.reason == "first_reason"

        for index in range(256):
            mark_terminated(
                f"grant-first-writer-churn-{index}",
                reason="other_reason",
                source="server",
            )
        assert run_termination(RUN_ID) is None
        replay = base._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert (replay.deny_source, replay.deny_reason) == (
            "sticky_replay",
            "first_reason",
        )

    def test_local_first_writer_is_not_relabelled_by_grant_directive(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        mark_terminated(
            RUN_ID,
            reason="velocity:repeat_size",
            source="local_velocity",
        )
        response = LeaseGrantResponse.model_validate(
            _grant_payload(
                directive_run_id=RUN_ID,
                directive_reason="manual_kill",
            )
        )

        outcome, _ = base._apply_lease_response(RUN_ID, response, ["gpt-5.5"])

        assert outcome is GrantOutcome.DENIED
        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.source, termination.reason) == (
            "local_velocity",
            "velocity:repeat_size",
        )
        assert response.run_control is not None
        assert response.run_control.reason == "manual_kill"
        sticky = base._run_hard_deny_responses[RUN_ID]
        assert sticky.run_control is not None
        assert sticky.run_control.reason == "manual_kill"

    def test_wrong_run_grant_directive_warns_but_installs_ordinary_grant(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        response = LeaseGrantResponse.model_validate(_grant_payload(directive_run_id=OTHER_RUN_ID))

        with caplog.at_level(logging.WARNING, logger="solwyn.budget"):
            outcome, _ = base._apply_lease_response(RUN_ID, response, ["gpt-5.5"])

        assert outcome is GrantOutcome.APPLIED
        assert base._lease.lease_id_for(RUN_ID) == "lse_a"
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        assert base._build_prior_hard_deny_unavailable_result(RUN_ID) is None
        assert sum("run_control.directive_run_mismatch" in r.message for r in caplog.records) == 1

    def test_wrong_run_denied_grant_enforces_ordinary_denial_without_reason_leak(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = BudgetEnforcer(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        payload = _grant_payload(directive_run_id=OTHER_RUN_ID)
        payload.update(
            {
                "allowed": False,
                "denied_by_period": "agent_run",
                "lease_id": None,
                "generation": None,
                "granted_tokens": None,
                "refresh_interval_s": None,
                "lease_length_s": None,
                "headroom_share_tokens": None,
                "posture": None,
                "final_grant": None,
            }
        )

        with (
            caplog.at_level(logging.WARNING, logger="solwyn.budget"),
            patch.object(enforcer._http, "post", return_value=_response(payload)),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                estimated_output_bound=100,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
                call_id=call_uuid("wrong-run-denied-grant"),
            )

        assert (result.allowed, result.denied_by_period, result.deny_reason) == (
            False,
            "agent_run",
            "agent_run",
        )
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is not None
        assert sum("run_control.directive_run_mismatch" in r.message for r in caplog.records) == 1

    def test_wrong_run_control_denied_grant_uses_outage_posture_without_state(self) -> None:
        enforcer = BudgetEnforcer(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        payload = _grant_payload(directive_run_id=OTHER_RUN_ID)
        payload.update(
            {
                "allowed": False,
                "denied_by_period": "run_stopped",
                "lease_id": None,
                "generation": None,
                "granted_tokens": None,
                "refresh_interval_s": None,
                "lease_length_s": None,
                "headroom_share_tokens": None,
                "posture": None,
                "final_grant": None,
            }
        )

        with patch.object(enforcer._http, "post", return_value=_response(payload)):
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                estimated_output_bound=100,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=RUN_ID,
                call_id=call_uuid("wrong-run-control-denied-grant"),
            )

        assert result.allowed is True
        assert result.denied_by_period is None
        assert enforcer._lease.lease_id_for(RUN_ID) is None
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        assert enforcer._build_prior_hard_deny_unavailable_result(RUN_ID) is None

    def test_fenced_wrong_run_grant_has_no_warning_or_state_effect(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        response = LeaseGrantResponse.model_validate(_grant_payload(directive_run_id=OTHER_RUN_ID))
        base._closed = True
        base._close_epoch = 1

        with caplog.at_level(logging.WARNING, logger="solwyn.budget"):
            outcome, surrender = base._apply_lease_response(
                RUN_ID,
                response,
                ["gpt-5.5"],
                close_epoch=0,
            )

        assert outcome is GrantOutcome.STALE
        assert surrender is not None
        assert base._lease.lease_id_for(RUN_ID) is None
        assert run_termination(RUN_ID) is None
        assert not any("run_control.directive_run_mismatch" in r.message for r in caplog.records)

    def test_current_renewal_directive_marks_and_drops_but_fenced_one_is_discarded(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        first = LeaseGrantResponse.model_validate(_grant_payload())
        assert base._apply_lease_response(RUN_ID, first, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert request is not None
        directed = LeaseGrantResponse.model_validate(
            _grant_payload(generation=2, directive_run_id=RUN_ID)
        )

        base._finish_renewal(RUN_ID, request, directed, ["gpt-5.5"])

        assert base._lease.lease_id_for(RUN_ID) is None
        assert run_termination(RUN_ID) is not None

        # A response originating from old A must not affect replacement B.
        _run_control.clear_run_termination(RUN_ID)
        base._lease.drop(RUN_ID)
        replacement = LeaseGrantResponse.model_validate(
            _grant_payload(lease_id="lse_b", generation=1)
        )
        assert (
            base._apply_lease_response(RUN_ID, replacement, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        )
        base._finish_renewal(RUN_ID, request, directed, ["gpt-5.5"])

        assert base._lease.lease_id_for(RUN_ID) == "lse_b"
        assert run_termination(RUN_ID) is None

    def test_repeated_renewal_directive_keeps_first_server_reason_everywhere(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        initial = LeaseGrantResponse.model_validate(_grant_payload())
        assert base._apply_lease_response(RUN_ID, initial, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        first_request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert first_request is not None
        first = LeaseGrantResponse.model_validate(
            _grant_payload(
                generation=2,
                directive_run_id=RUN_ID,
                directive_reason="first_reason",
            )
        )
        base._finish_renewal(RUN_ID, first_request, first, ["gpt-5.5"])

        replacement = LeaseGrantResponse.model_validate(
            _grant_payload(lease_id="lse_b", generation=1)
        )
        assert (
            base._apply_lease_response(RUN_ID, replacement, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        )
        second_request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert second_request is not None
        second = LeaseGrantResponse.model_validate(
            _grant_payload(
                lease_id="lse_b",
                generation=2,
                directive_run_id=RUN_ID,
                directive_reason="second_reason",
            )
        )
        base._finish_renewal(RUN_ID, second_request, second, ["gpt-5.5"])

        termination = run_termination(RUN_ID)
        assert termination is not None
        assert (termination.source, termination.reason) == ("server", "first_reason")
        assert second.run_control is not None
        assert second.run_control.reason == "first_reason"
        sticky = base._run_hard_deny_responses[RUN_ID]
        assert sticky.run_control is not None
        assert sticky.run_control.reason == "first_reason"

        for index in range(256):
            mark_terminated(
                f"renew-first-writer-churn-{index}",
                reason="other_reason",
                source="server",
            )
        assert run_termination(RUN_ID) is None
        replay = base._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert replay.deny_reason == "first_reason"

    def test_renewal_directive_cannot_drop_a_replacement_installed_after_its_fence(
        self,
    ) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        first = LeaseGrantResponse.model_validate(_grant_payload())
        assert base._apply_lease_response(RUN_ID, first, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert request is not None
        directed = LeaseGrantResponse.model_validate(
            _grant_payload(generation=2, directive_run_id=RUN_ID)
        )
        replacement = LeaseGrantResponse.model_validate(
            _grant_payload(lease_id="lse_b", generation=1)
        )
        mark_entered = threading.Event()
        release_mark = threading.Event()
        replacement_done = threading.Event()
        real_mark = _run_control._mark_terminated_locked

        def blocked_mark(run_id: str, *, reason: str, source: str) -> object:
            mark_entered.set()
            assert release_mark.wait(timeout=2)
            return real_mark(run_id, reason=reason, source=source)  # type: ignore[arg-type]

        def finish_directed_renewal() -> None:
            base._finish_renewal(RUN_ID, request, directed, ["gpt-5.5"])

        def install_replacement() -> None:
            with base._state_lock:
                base._lease.drop(RUN_ID)
            outcome, _ = base._apply_lease_response(RUN_ID, replacement, ["gpt-5.5"])
            assert outcome is GrantOutcome.APPLIED
            replacement_done.set()

        with patch.object(
            _run_control,
            "_mark_terminated_locked",
            side_effect=blocked_mark,
        ):
            finisher = threading.Thread(target=finish_directed_renewal)
            finisher.start()
            assert mark_entered.wait(timeout=2)
            installer = threading.Thread(target=install_replacement)
            installer.start()
            replacement_done.wait(timeout=0.2)
            release_mark.set()
            finisher.join(timeout=2)
            installer.join(timeout=2)

        assert not finisher.is_alive()
        assert not installer.is_alive()
        assert replacement_done.is_set()
        assert base._lease.lease_id_for(RUN_ID) == "lse_b"

    def test_current_wrong_run_renewal_warns_but_installs_normally(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        first = LeaseGrantResponse.model_validate(_grant_payload())
        assert base._apply_lease_response(RUN_ID, first, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert request is not None
        wrong = LeaseGrantResponse.model_validate(
            _grant_payload(generation=2, directive_run_id=OTHER_RUN_ID)
        )

        with caplog.at_level(logging.WARNING, logger="solwyn.budget"):
            base._finish_renewal(RUN_ID, request, wrong, ["gpt-5.5"])

        state = base._lease.state_for(RUN_ID)
        assert state is not None
        assert (state.lease_id, state.generation) == ("lse_a", 2)
        assert state.renewal_in_flight is False
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        assert sum("run_control.directive_run_mismatch" in r.message for r in caplog.records) == 1

    def test_current_wrong_run_denied_renewal_applies_ordinary_denial(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        first = LeaseGrantResponse.model_validate(_grant_payload())
        assert base._apply_lease_response(RUN_ID, first, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert request is not None
        payload = _grant_payload(generation=2, directive_run_id=OTHER_RUN_ID)
        payload.update(
            {
                "allowed": False,
                "denied_by_period": "agent_run",
                "lease_id": None,
                "generation": None,
                "granted_tokens": None,
                "refresh_interval_s": None,
                "lease_length_s": None,
                "headroom_share_tokens": None,
                "posture": None,
                "final_grant": None,
            }
        )
        denied = LeaseGrantResponse.model_validate(payload)

        base._finish_renewal(RUN_ID, request, denied, ["gpt-5.5"])

        assert base._lease.lease_id_for(RUN_ID) is None
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        replay = base._build_prior_hard_deny_unavailable_result(RUN_ID)
        assert replay is not None
        assert (replay.denied_by_period, replay.deny_reason) == ("agent_run", "agent_run")

    def test_current_wrong_run_control_denied_renewal_retains_authority(self) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        first = LeaseGrantResponse.model_validate(_grant_payload())
        assert base._apply_lease_response(RUN_ID, first, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert request is not None
        payload = _grant_payload(generation=2, directive_run_id=OTHER_RUN_ID)
        payload.update(
            {
                "allowed": False,
                "denied_by_period": "run_stopped",
                "lease_id": None,
                "generation": None,
                "granted_tokens": None,
                "refresh_interval_s": None,
                "lease_length_s": None,
                "headroom_share_tokens": None,
                "posture": None,
                "final_grant": None,
            }
        )

        base._finish_renewal(
            RUN_ID,
            request,
            LeaseGrantResponse.model_validate(payload),
            ["gpt-5.5"],
        )

        state = base._lease.state_for(RUN_ID)
        assert state is not None
        assert (state.lease_id, state.generation) == ("lse_a", 1)
        assert state.renewal_in_flight is False
        assert run_termination(RUN_ID) is None
        assert run_termination(OTHER_RUN_ID) is None
        assert base._build_prior_hard_deny_unavailable_result(RUN_ID) is None

    def test_fenced_wrong_run_renewal_has_no_warning_or_state_effect(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        base = _BudgetEnforcerBase("https://api.test.solwyn.ai", VALID_API_KEY)
        first = LeaseGrantResponse.model_validate(_grant_payload())
        assert base._apply_lease_response(RUN_ID, first, ["gpt-5.5"])[0] is GrantOutcome.APPLIED
        request = base._build_renewal(
            RUN_ID,
            model="gpt-5.5",
            provider="openai",
            fallback_providers=[],
            fallback_models=[],
        )
        assert request is not None
        with base._state_lock:
            base._lease.drop(RUN_ID)
            replacement = LeaseGrantResponse.model_validate(
                _grant_payload(lease_id="lse_b", generation=1)
            )
            assert (
                base._lease.apply_grant_response(
                    RUN_ID,
                    replacement,
                    now=budget_module.time.monotonic(),
                    declared_models=["gpt-5.5"],
                )
                is GrantOutcome.APPLIED
            )
        wrong = LeaseGrantResponse.model_validate(
            _grant_payload(generation=2, directive_run_id=OTHER_RUN_ID)
        )

        with caplog.at_level(logging.WARNING, logger="solwyn.budget"):
            base._finish_renewal(RUN_ID, request, wrong, ["gpt-5.5"])

        state = base._lease.state_for(RUN_ID)
        assert state is not None
        assert (state.lease_id, state.generation) == ("lse_b", 1)
        assert run_termination(RUN_ID) is None
        assert not any("run_control.directive_run_mismatch" in r.message for r in caplog.records)
