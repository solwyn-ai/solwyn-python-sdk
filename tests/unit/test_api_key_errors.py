"""API key scoping: the refusal matrix F1 to F18 (core spec section 6.6).

Each fixture is one named test (``test_fNN_...``). Most run end to end: a real
``Solwyn`` / ``AsyncSolwyn`` wrapper over a stub provider client, talking to
the ``FakeControlPlane`` double, which implements core's key path (the
``Solwyn-Project`` header rules, ``GET /api/v1/auth/key``, and the section 2.4
refusal codes and bodies). The fixtures that need two requests interleaved in
an exact order (F12, F16, F17) drive a ``BudgetEnforcer`` through a scripted
transport instead, so the ordering is deterministic rather than timed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID, auth_key_body, validated_binding

from solwyn._constants import AUTH_KEY_PATH, PROJECT_HEADER
from solwyn._project import ProjectBinding
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ApiKeyRefusedError, ConfigurationError
from solwyn.testing import FakeBudgetRule, FakeControlPlane

# Every test here runs the gate's real automatic validation.
pytestmark = pytest.mark.project_validation

_CHECK_PATH = "/api/v1/budgets/check"
_CONFIRM_PATH = "/api/v1/budgets/confirm"
_INGEST_PATH = "/api/v1/metadata/ingest"
_URL = "https://api.test.solwyn.ai"

_READ_ONLY_MESSAGE = (
    "This Solwyn API key is read-only. The SDK reports spending, so it needs a "
    "full-scope key. Create one in the dashboard or with solwyn keys create."
)
_INVALID_MESSAGE = "The Solwyn API key was not recognised or has been revoked."


# ── provider stubs ────────────────────────────────────────────────────────


class _Completions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))


class _OpenAIStub:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_Completions())

    def with_options(self, **_kwargs: object) -> _OpenAIStub:
        return self


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


class _AsyncCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))


class _AsyncOpenAIStub:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_AsyncCompletions())

    def with_options(self, **_kwargs: object) -> _AsyncOpenAIStub:
        return self


_AsyncOpenAIStub.__module__ = "openai._client"
_AsyncOpenAIStub.__name__ = "AsyncOpenAI"


class _Clock:
    """A settable monotonic clock for the binding's revalidation schedule."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@contextmanager
def _wrapped(
    plane: FakeControlPlane, **kwargs: Any
) -> Iterator[tuple[Solwyn, _OpenAIStub, _Clock]]:
    stub = _OpenAIStub()
    options: dict[str, Any] = {
        "lease_enabled": False,
        "fail_open": True,
        "report_untracked_surfaces": False,
        "breaker_reporting_enabled": False,
    }
    options.update(kwargs)
    client = plane.wrap(stub, **options)
    clock = _Clock()
    client._solwyn_binding._clock = clock
    try:
        yield client, stub, clock
    finally:
        client.close()


def _call(client: Solwyn) -> object:
    return client.chat.completions.create(
        model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
    )


def _record_effects(binding: ProjectBinding) -> list[tuple[str, str]]:
    """Every (code, effect) the binding records, in order."""
    effects: list[tuple[str, str]] = []

    def listener() -> None:
        refusal = binding.refusal
        if refusal is not None:
            effects.append((refusal.code, refusal.effect))

    binding.add_listener(listener)
    return effects


def _requests_to(plane: FakeControlPlane, path: str) -> int:
    return sum(1 for seen, _ in plane.project_headers if seen == path)


def _expired_401(posture: object = "alert_only", *, omit: bool = False) -> bytes:
    detail: dict[str, Any] = {
        "code": "api_key_expired",
        "message": "This API key expired on 2026-11-01.",
        "expired_at": "2026-11-01T00:00:00Z",
        "api_key_id": "00000000-0000-4000-8000-00000000000e",
    }
    if not omit:
        detail["project_budget_mode"] = posture
    return json.dumps({"detail": detail}).encode()


# ── F1 to F2: read-only keys ──────────────────────────────────────────────


@pytest.mark.unit
def test_f01_validate_with_read_only_key_raises_configuration_error() -> None:
    plane = FakeControlPlane()
    plane.set_key_scope("read_only")
    with _wrapped(plane) as (client, stub, _clock):
        with pytest.raises(ConfigurationError) as excinfo:
            client.validate()
        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert isinstance(excinfo.value, ApiKeyRefusedError)
    assert excinfo.value.code == "read_only_key"
    assert excinfo.value.field == "api_key"
    assert str(excinfo.value) == _READ_ONLY_MESSAGE
    assert stub.chat.completions.calls == 0
    assert plane.checks == []


@pytest.mark.unit
def test_f02_runtime_read_only_key_blocks_despite_fail_open_and_stays_blocked() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane, fail_open=True) as (client, stub, _clock):
        client.validate()
        _call(client)
        plane.set_key_scope("read_only")
        before = _requests_to(plane, _CHECK_PATH)

        with pytest.raises(ApiKeyRefusedError) as first:
            _call(client)
        after_first = _requests_to(plane, _CHECK_PATH)
        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert first.value.code == "read_only_key"
    assert stub.chat.completions.calls == 1
    # One refused check; the next call never left the gate.
    assert after_first == before + 1
    assert _requests_to(plane, _CHECK_PATH) == after_first


# ── F3 to F6b: expired / revoked keys and the posture rule ────────────────


@pytest.mark.unit
def test_f03_alert_only_then_hard_deny_then_expiry_blocks() -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _wrapped(plane) as (client, stub, _clock):
        assert client.validate() is not None
        plane.mode = plane.mode.__class__("hard_deny")
        plane.expire_key()

        with pytest.raises(ApiKeyRefusedError) as excinfo:
            _call(client)

    assert excinfo.value.code == "api_key_expired"
    assert excinfo.value.field == "api_key"
    expired_on = plane.key_expires_at.date().isoformat()
    assert str(excinfo.value) == (
        f"The Solwyn API key expired on {expired_on}. Rotate it in the dashboard or "
        "with solwyn keys rotate."
    )
    assert client._solwyn_binding.refusal is not None
    assert client._solwyn_binding.refusal.posture == "hard_deny"
    assert stub.chat.completions.calls == 0


@pytest.mark.unit
def test_f04_hard_deny_then_alert_only_then_expiry_runs_unmetered_with_error_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane) as (client, stub, _clock):
        client.validate()
        effects = _record_effects(client._solwyn_binding)
        plane.mode = plane.mode.__class__("alert_only")
        plane.expire_key()

        with caplog.at_level(logging.ERROR):
            _call(client)
            checks_after_first = len(plane.checks)
            _call(client)

    assert stub.chat.completions.calls == 2
    # The check's 401 status blocked at once (its own alert_only body never
    # relaxes it); the immediate revalidation, sent by the same call, answered
    # alert_only and relaxed it, so the call ran unmetered.
    assert effects[:2] == [("api_key_invalid", "block"), ("api_key_expired", "allow_unmetered")]
    assert len(plane.auth_key_requests) >= 2
    # Neither call kept a check or a reservation.
    assert checks_after_first == len(plane.checks) == 0
    assert caplog.text.count("solwyn.api_key_refused") >= 2


@pytest.mark.unit
def test_f05_alert_only_project_with_unpaused_scoped_hard_rule_blocks() -> None:
    plane = FakeControlPlane(mode="alert_only")
    plane.budget_rules.append(FakeBudgetRule("scoped", "hard_deny"))
    with _wrapped(plane) as (client, stub, _clock):
        client.validate()
        plane.expire_key()

        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert plane.refusal_posture() == "hard_deny"
    assert stub.chat.completions.calls == 0


@pytest.mark.unit
def test_f05b_paused_scoped_hard_rule_allows_unmetered() -> None:
    plane = FakeControlPlane(mode="alert_only")
    plane.budget_rules.append(FakeBudgetRule("scoped", "hard_deny", paused=True))
    with _wrapped(plane) as (client, stub, _clock):
        client.validate()
        plane.expire_key()
        _call(client)

    assert plane.refusal_posture() == "alert_only"
    assert stub.chat.completions.calls == 1


@pytest.mark.unit
@pytest.mark.parametrize("snoozed", [False, True])
def test_f05c_unpaused_hard_run_cap_blocks_snoozed_or_not(snoozed: bool) -> None:
    plane = FakeControlPlane(mode="alert_only")
    plane.budget_rules.append(FakeBudgetRule("runaway_run", "hard_deny", snoozed=snoozed))
    with _wrapped(plane) as (client, stub, _clock):
        client.validate()
        plane.expire_key()

        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert plane.refusal_posture() == "hard_deny"
    assert stub.chat.completions.calls == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "rule",
    [
        FakeBudgetRule("runaway_run", "hard_deny", paused=True),
        FakeBudgetRule("runaway_run", "alert_only"),
    ],
    ids=["paused", "alert_only"],
)
def test_f05d_paused_or_alert_only_run_cap_allows_unmetered(rule: FakeBudgetRule) -> None:
    plane = FakeControlPlane(mode="alert_only")
    plane.budget_rules.append(rule)
    with _wrapped(plane) as (client, stub, _clock):
        client.validate()
        plane.expire_key()
        _call(client)

    assert plane.refusal_posture() == "alert_only"
    assert stub.chat.completions.calls == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [_expired_401(None), _expired_401(omit=True), _expired_401("soft_deny")],
    ids=["null", "missing", "unknown"],
)
def test_f06_401_without_a_known_posture_blocks(body: bytes) -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _wrapped(plane) as (client, stub, _clock):
        client.validate()
        with plane.raw_response(401, body, path=_CHECK_PATH), pytest.raises(ApiKeyRefusedError):
            _call(client)

    refusal = client._solwyn_binding.refusal
    assert refusal is not None
    assert (refusal.code, refusal.posture, refusal.effect) == (
        "api_key_expired",
        "unknown",
        "block",
    )
    assert stub.chat.completions.calls == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b"",
        json.dumps({"detail": "Invalid or revoked API key"}).encode(),
        json.dumps({"detail": {"message": "x"}}).encode(),
        json.dumps({"detail": {"code": "something_new"}}).encode(),
    ],
    ids=["invalid_json", "empty", "string_detail", "no_code", "unknown_code"],
)
def test_f06b_any_401_body_is_an_invalid_key_refusal_never_an_outage(
    body: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _wrapped(plane, fail_open=True, control_plane_failure_threshold=1) as (
        client,
        stub,
        _clock,
    ):
        client.validate()
        with (
            caplog.at_level(logging.WARNING),
            plane.raw_response(401, body, path=_CHECK_PATH),
            pytest.raises(ApiKeyRefusedError) as excinfo,
        ):
            _call(client)
        breaker_state = client._solwyn_control_plane_breaker.get_state().state

    refusal = client._solwyn_binding.refusal
    assert refusal is not None
    assert (refusal.code, refusal.posture, refusal.effect) == (
        "api_key_invalid",
        "unknown",
        "block",
    )
    assert str(excinfo.value) == _INVALID_MESSAGE
    # Never the outage ladder: no outage warning, and the breaker stays closed.
    assert "Cloud API budget check failed" not in caplog.text
    assert breaker_state.value == "closed"
    assert stub.chat.completions.calls == 0


# ── F7 to F11: persistence of a refusal ───────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["outage", "5xx"])
def test_f07_block_survives_failed_revalidation_and_outage_exits(failure: str) -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane, fail_open=True) as (client, stub, clock):
        client.validate()
        plane.revoke_key()
        with pytest.raises(ApiKeyRefusedError):
            _call(client)
        plane.restore_key()  # only a fresh 200 may clear it; none arrives

        for _ in range(3):
            clock.advance(61)
            window = (
                plane.outage() if failure == "outage" else plane.raw_response(503, b"unavailable")
            )
            with window, pytest.raises(ApiKeyRefusedError):
                _call(client)

    assert client._solwyn_binding.state == "refused"
    assert stub.chat.completions.calls == 0


@pytest.mark.unit
def test_f07_block_survives_revalidation_timeouts_and_check_timeouts() -> None:
    clock = _Clock()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if len(seen) == 1:
            return httpx.Response(401, content=_HARD_401, request=request)
        raise httpx.ReadTimeout("scripted", request=request)

    binding = validated_binding()
    binding._clock = clock
    enforcer = BudgetEnforcer(
        _URL,
        binding,
        fail_open=True,
        cache_ttl=0,
        lease_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        for _ in range(4):
            with pytest.raises(ApiKeyRefusedError):
                enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
            clock.advance(61)
    finally:
        enforcer.close()

    # One refused check; its 401 scheduled an immediate revalidation, then
    # only the 60 s revalidations, each timing out.
    assert seen == [_CHECK_PATH] + [AUTH_KEY_PATH] * 4
    assert binding.state == "refused"


@pytest.mark.unit
def test_f07b_allow_unmetered_survives_repeated_revalidation_without_raising() -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _wrapped(plane) as (client, stub, clock):
        client.validate()
        plane.expire_key()
        for _ in range(4):
            _call(client)
            clock.advance(61)

    # No call ever raised; each ran unmetered. The first call's 401 was
    # revalidated at once, then every 60 s; each answer was the same 401.
    assert stub.chat.completions.calls == 4
    assert len(plane.auth_key_requests) >= 1 + 1 + 3


@pytest.mark.unit
def test_f08_refusal_beats_a_held_lease_and_a_cached_allow() -> None:
    import solwyn

    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane, lease_enabled=True, budget_check_cache_ttl=60) as (
        client,
        stub,
        _clock,
    ):
        client.validate()
        _call(client)  # caches an allow
        with solwyn.run("f08-run"):
            _call(client)  # grants a lease
            assert len(client._solwyn_budget._lease.active_run_ids()) == 1
            plane.revoke_key()
            with pytest.raises(ApiKeyRefusedError):
                client.validate()
            requests_before = len(plane.project_headers)
            with pytest.raises(ApiKeyRefusedError):
                _call(client)  # would draw on the held lease
        with pytest.raises(ApiKeyRefusedError):
            _call(client)  # would replay the cached allow

    assert stub.chat.completions.calls == 2
    # The held lease was dropped locally. Neither blocked call asked for a
    # check or a lease (background settlement of the two earlier calls may
    # still have been sent, and refused).
    assert client._solwyn_budget._lease.active_run_ids() == []
    late = {path for path, _ in plane.project_headers[requests_before:]}
    assert not late & {_CHECK_PATH, "/api/v1/budgets/lease", "/api/v1/budgets/lease/renew"}


@pytest.mark.unit
def test_f09_refusal_beats_an_open_breaker() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(
        plane,
        fail_open=True,
        control_plane_failure_threshold=1,
        control_plane_recovery_timeout=600,
    ) as (client, stub, _clock):
        client.validate()
        with plane.outage(path=_CHECK_PATH):
            _call(client)  # fail-open; opens the breaker
        assert client._solwyn_control_plane_breaker.get_state().state.value == "open"
        plane.revoke_key()
        with pytest.raises(ApiKeyRefusedError):
            client.validate()

        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert stub.chat.completions.calls == 1


@pytest.mark.unit
def test_f10_revalidation_200_recovers_and_the_header_is_the_pinned_id() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane) as (client, stub, clock):
        plane.revoke_key()
        with pytest.raises(ApiKeyRefusedError):
            _call(client)
        plane.restore_key()
        clock.advance(61)

        _call(client)
        _call(client)

    assert client._solwyn_binding.state == "valid"
    assert client._solwyn_binding.pinned_id == plane.project_id
    check_headers = [value for path, value in plane.project_headers if path == _CHECK_PATH]
    assert check_headers[-2:] == [plane.project_id, plane.project_id]
    assert stub.chat.completions.calls == 2


@pytest.mark.unit
@pytest.mark.parametrize("path", [_CONFIRM_PATH, _INGEST_PATH])
def test_f11_project_not_found_on_background_send_drops_item_and_blocks_next_call(
    path: str,
) -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane, reporter_flush_interval=3600) as (client, stub, _clock):
        client.validate()
        _call(client)
        plane.archive_project()
        reporter = client._solwyn_reporter
        reporter._flush_remaining()
        sends = _requests_to(plane, path)
        reporter._flush_remaining()  # nothing left to retry

        with pytest.raises(ApiKeyRefusedError) as excinfo:
            _call(client)

    assert sends >= 1
    assert _requests_to(plane, path) == sends
    assert plane.confirms == [] and plane.ingested == []
    assert len(reporter._confirm_queue) == len(reporter._settlement_queue) == 0
    assert len(reporter._queue) == 0
    assert excinfo.value.code == "project_not_found"
    assert excinfo.value.field == "project"
    assert str(excinfo.value) == (
        f"Solwyn project '{plane.project_name}' was not found for this API key. Check "
        "SOLWYN_PROJECT. A renamed project's old name stops working; its proj_ id "
        "always works."
    )
    assert stub.chat.completions.calls == 1


# ── F12: every exit of an in-flight check sees a newer refusal ────────────


def _check_body(*, allowed: bool = True) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "remaining_budget": 50.0,
        "reservation_id": "res-b",
        "mode": "hard_deny",
        "budget_limit": 100.0,
        "current_usage": 50.0,
        "project_id": VALID_PROJECT_ID,
    }


_HARD_401 = _expired_401("hard_deny")
_ALERT_401 = _expired_401("alert_only")


def _b_outcome(variant: str, request: httpx.Request) -> httpx.Response:
    if variant == "allow":
        return httpx.Response(200, json=_check_body(), request=request)
    if variant == "timeout":
        raise httpx.ReadTimeout("scripted", request=request)
    if variant == "5xx":
        return httpx.Response(503, json={"detail": "unavailable"}, request=request)
    if variant == "429":
        return httpx.Response(
            429, json={"detail": "slow down"}, headers={"Retry-After": "1"}, request=request
        )
    if variant == "unreadable":
        return httpx.Response(200, content=b"<html>", request=request)
    raise httpx.ConnectError("scripted", request=request)


_F12_VARIANTS = ["allow", "timeout", "5xx", "429", "unreadable", "transport_error"]


@pytest.mark.unit
@pytest.mark.parametrize("variant", _F12_VARIANTS)
def test_f12_call_b_is_blocked_on_every_exit_after_call_a_records_a_hard_401(
    variant: str,
) -> None:
    enforcer: BudgetEnforcer | None = None
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if len(seen) == 1:
            # B is in flight (it passed the gate). A runs start to finish now
            # and records a hard 401 before B's answer arrives.
            if enforcer is None:
                raise RuntimeError("enforcer not built")
            with pytest.raises(ApiKeyRefusedError):
                enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
            return _b_outcome(variant, request)
        return httpx.Response(401, content=_HARD_401, request=request)

    enforcer = BudgetEnforcer(
        _URL,
        validated_binding(),
        fail_open=True,
        cache_ttl=0,
        lease_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ApiKeyRefusedError) as excinfo:
            enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()

    assert excinfo.value.code == "api_key_expired"
    # B's check, A's check, and A's immediate revalidation (refused again).
    assert seen == [_CHECK_PATH, _CHECK_PATH, AUTH_KEY_PATH]
    # The fail-open builder did not tally B as an admitted outage call.
    assert variant == "allow" or enforcer.uncounted_tally()[0] <= 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("variant", _F12_VARIANTS)
async def test_f12_async_call_b_is_blocked_on_every_exit(variant: str) -> None:
    binding = validated_binding()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if len(seen) == 1:
            # A's answer lands while B waits: record A's hard 401 directly
            # (the async enforcer cannot be re-entered from the transport).
            _headers, dispatched_at = binding.request_headers()
            binding.apply_auth_key(401, _HARD_401, dispatched_at)
            return _b_outcome(variant, request)
        return httpx.Response(401, content=_HARD_401, request=request)

    enforcer = AsyncBudgetEnforcer(
        _URL,
        binding,
        fail_open=True,
        cache_ttl=0,
        lease_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ApiKeyRefusedError):
            await enforcer.check_budget(
                estimated_input_tokens=1, model="gpt-5.5", provider="openai"
            )
    finally:
        await enforcer.close()

    assert seen == [_CHECK_PATH]


# ── F13 to F15: startup ───────────────────────────────────────────────────


@pytest.mark.unit
def test_f13_startup_expired_key_blocks_even_when_alert_only() -> None:
    plane = FakeControlPlane(mode="alert_only")
    plane.expire_key()
    with _wrapped(plane) as (client, stub, _clock):
        with pytest.raises(ApiKeyRefusedError) as first:
            _call(client)
        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert first.value.code == "api_key_expired"
    refusal = client._solwyn_binding.refusal
    assert refusal is not None
    assert (refusal.posture, refusal.effect) == ("alert_only", "block")
    assert client._solwyn_binding.ever_valid is False
    # One automatic validation; no check was ever sent.
    assert len(plane.auth_key_requests) == 1
    assert plane.checks == []
    assert stub.chat.completions.calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_f13_async_startup_expired_key_blocks() -> None:
    plane = FakeControlPlane(mode="alert_only")
    plane.expire_key()
    stub = _AsyncOpenAIStub()
    client: AsyncSolwyn = plane.wrap_async(stub, lease_enabled=False)
    try:
        for _ in range(2):
            with pytest.raises(ApiKeyRefusedError):
                await client.chat.completions.create(model="gpt-5.5", messages=[])
    finally:
        await client.close()

    assert len(plane.auth_key_requests) == 1
    assert stub.chat.completions.calls == 0


@pytest.mark.unit
def test_f14_startup_outage_then_recovery_pins_the_id_and_survives_a_rename() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane) as (client, stub, clock):
        with plane.outage(path=AUTH_KEY_PATH):
            _call(client)
            _call(client)  # no new validation before 30 s
        assert client._solwyn_binding.state == "unvalidated"
        unvalidated_checks = [v for p, v in plane.project_headers if p == _CHECK_PATH]

        clock.advance(31)
        _call(client)
        plane.rename_project("checkout-bot-renamed")
        _call(client)

    assert unvalidated_checks == [plane_name := "fake-project", plane_name]
    assert client._solwyn_binding.pinned_id == plane.project_id
    check_headers = [v for p, v in plane.project_headers if p == _CHECK_PATH]
    assert check_headers[-2:] == [plane.project_id, plane.project_id]
    assert len(plane.auth_key_requests) == 1  # outage attempts never reached auth
    assert stub.chat.completions.calls == 4


@pytest.mark.unit
def test_f15_unvalidated_check_404_project_not_found_refuses_and_blocks() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane) as (client, stub, _clock):
        plane.rename_project("someone-renamed-it")
        with plane.outage(path=AUTH_KEY_PATH), pytest.raises(ApiKeyRefusedError) as first:
            _call(client)
        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert first.value.code == "project_not_found"
    assert client._solwyn_binding.state == "refused"
    assert stub.chat.completions.calls == 0


# ── F16 to F18: fencing ───────────────────────────────────────────────────


@pytest.mark.unit
def test_f16_a_delayed_alert_only_401_never_weakens_a_newer_hard_401() -> None:
    enforcer: BudgetEnforcer | None = None
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if len(seen) == 1:
            # A is in flight; the owner enables a hard rule; B's lookup sees it.
            if enforcer is None:
                raise RuntimeError("enforcer not built")
            with pytest.raises(ApiKeyRefusedError):
                enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
            # A's lookup happened before the rule: its delayed answer says alert_only.
            return httpx.Response(401, content=_ALERT_401, request=request)
        return httpx.Response(401, content=_HARD_401, request=request)

    binding = validated_binding()
    enforcer = BudgetEnforcer(
        _URL, binding, cache_ttl=0, lease_enabled=False, transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ApiKeyRefusedError):
            enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
        with pytest.raises(ApiKeyRefusedError):
            enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()

    refusal = binding.refusal
    assert refusal is not None
    assert (refusal.posture, refusal.effect) == ("hard_deny", "block")
    # B's revalidation answered hard too; the third call never left the gate.
    assert seen == [_CHECK_PATH, _CHECK_PATH, AUTH_KEY_PATH]


@pytest.mark.unit
def test_f17_a_delayed_auth_key_200_never_clears_a_newer_refusal() -> None:
    enforcer: BudgetEnforcer | None = None
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == AUTH_KEY_PATH and seen.count(AUTH_KEY_PATH) == 1:
            # validate() is in flight; check C runs and records a hard 401.
            if enforcer is None:
                raise RuntimeError("enforcer not built")
            with pytest.raises(ApiKeyRefusedError):
                enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
            return httpx.Response(200, content=auth_key_body(), request=request)
        return httpx.Response(401, content=_HARD_401, request=request)

    binding = validated_binding()
    enforcer = BudgetEnforcer(
        _URL,
        binding,
        cache_ttl=0,
        lease_enabled=False,
        transport=httpx.MockTransport(handler),
        validation_timeout=0.05,
    )
    try:
        with pytest.raises(ApiKeyRefusedError):
            enforcer.validate()
        with pytest.raises(ApiKeyRefusedError):
            enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")
    finally:
        enforcer.close()

    assert binding.state == "refused"
    # C's immediate revalidation joined validate()'s flight; the 200 that flight
    # got was fenced, so the next call revalidated again (refused).
    assert seen == [AUTH_KEY_PATH, _CHECK_PATH, AUTH_KEY_PATH]


@pytest.mark.unit
def test_f18_a_fresh_revalidation_may_relax_block_to_allow_unmetered() -> None:
    plane = FakeControlPlane(mode="alert_only")
    rule = FakeBudgetRule("scoped", "hard_deny")
    plane.budget_rules.append(rule)
    with _wrapped(plane) as (client, stub, clock):
        client.validate()
        plane.expire_key()
        with pytest.raises(ApiKeyRefusedError):
            _call(client)
        assert client._solwyn_binding.refusal is not None
        assert client._solwyn_binding.refusal.effect == "block"
        effects = _record_effects(client._solwyn_binding)

        rule.paused = True
        clock.advance(61)
        _call(client)

    # The revalidation dispatched after the stored refusal answered alert_only
    # and relaxed it; the call ran unmetered.
    assert effects[0] == ("api_key_expired", "allow_unmetered")
    assert stub.chat.completions.calls == 1


# ── header and binding wiring ─────────────────────────────────────────────


@pytest.mark.unit
def test_every_request_carries_the_header_and_validation_is_lazy() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _wrapped(plane) as (client, _stub, _clock):
        assert plane.project_headers == []  # no network in the constructor
        _call(client)
        client._solwyn_reporter._flush_remaining()

    paths = [path for path, _ in plane.project_headers]
    assert paths[0] == AUTH_KEY_PATH
    assert plane.project_headers[0][1] == "fake-project"
    assert all(value == plane.project_id for _, value in plane.project_headers[1:])
    assert {_CHECK_PATH, _CONFIRM_PATH} <= set(paths)


@pytest.mark.unit
def test_breaker_reports_wait_for_the_pinned_id() -> None:
    binding = ProjectBinding(VALID_API_KEY, "fake-project")
    plane = FakeControlPlane()
    with _wrapped(plane, breaker_reporting_enabled=True) as (client, _stub, _clock):
        assert client._solwyn_reporter._breaker_reporting_project_id() is None
        client.validate()
        assert client._solwyn_reporter._breaker_reporting_project_id() == plane.project_id
    assert binding.pinned_id is None


@pytest.mark.unit
def test_header_name_is_the_core_constant() -> None:
    assert PROJECT_HEADER == "Solwyn-Project"


@pytest.mark.unit
def test_failover_hop_recheck_refusal_raises_and_never_dispatches_the_fallback() -> None:
    """A hard_deny failover hop re-admits through check_budget, so it meets the gate too."""
    openai = pytest.importorskip("openai")
    plane = FakeControlPlane(mode="hard_deny")
    fallback_bodies: list[bytes] = []

    def primary(request: httpx.Request) -> httpx.Response:
        # The key is revoked while the primary attempt is in flight.
        plane.revoke_key()
        return httpx.Response(429, json={"error": {"message": "x", "type": "x", "code": None}})

    def fallback(request: httpx.Request) -> httpx.Response:
        fallback_bodies.append(request.content)
        raise AssertionError("the fallback must not be dispatched after a refusal")

    def client(handler: Callable[[httpx.Request], httpx.Response], base_url: str | None) -> Any:
        return openai.OpenAI(
            api_key="sk-test",
            base_url=base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )

    wrapped = plane.wrap(
        client(primary, None),
        fallback=[(client(fallback, "https://api.groq.com/openai/v1"), "llama-3.3-70b-versatile")],
        lease_enabled=False,
    )
    with wrapped, pytest.raises(ApiKeyRefusedError) as excinfo:
        wrapped.chat.completions.create(
            model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
        )

    assert excinfo.value.code == "api_key_invalid"
    assert fallback_bodies == []
    assert [check.model for check in plane.checks] == ["gpt-5.5"]


@pytest.mark.unit
def test_exit_drain_refusal_is_terminal_recorded_and_credits_the_breaker() -> None:
    from solwyn import _lifecycle
    from solwyn.circuit_breaker import CircuitBreaker

    binding = validated_binding()
    breaker = CircuitBreaker(name="control-plane", failure_threshold=1)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers[PROJECT_HEADER])
        return httpx.Response(404, json={"detail": {"code": "project_not_found"}})

    headers, dispatched_at = binding.request_headers()
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        outcome = _lifecycle._post_confirm_blocking(
            http,
            f"{_URL}{_CONFIRM_PATH}",
            {},
            headers,
            breaker,
            _lifecycle._monotonic() + 5,
            binding=binding,
            dispatched_at=dispatched_at,
        )

    assert outcome == "api_key_refused"
    assert seen == [VALID_PROJECT_ID]
    assert binding.refusal is not None and binding.refusal.code == "project_not_found"
    assert breaker.get_state().state.value == "closed"


@pytest.mark.unit
def test_lease_surrender_refusal_is_recorded_and_never_retried() -> None:
    binding = validated_binding()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(401, content=_HARD_401, request=request)

    enforcer = BudgetEnforcer(_URL, binding, transport=httpx.MockTransport(handler))
    try:
        from solwyn._types import LeaseSurrenderRequest

        request = LeaseSurrenderRequest(
            lease_id="lease-1", holder_id="holder-1", generation=1, spent_tokens=0
        )
        enforcer._surrender_payloads([request], deadline=_monotonic_now() + 5)
        counts = enforcer.release_counts
    finally:
        enforcer.close()

    assert calls == ["/api/v1/budgets/lease/surrender"]
    assert counts.get("refused") == 1
    assert binding.state == "refused"


def _monotonic_now() -> float:
    import time

    return time.monotonic()
