"""While the binding is refused, Solwyn hears nothing but /auth/key.

Rule: in ANY refused state (``block`` or ``allow_unmetered``) the SDK
dispatches no data-plane request: no check (the gate applies the stored
effect), no lease grant, renewal or surrender, no confirm, ingest, untracked or
breaker upload. Those items are dropped locally and counted as
``api_key_refused`` drops. The only traffic is the single-flight revalidation.

(a) an ``allow_unmetered`` client stays unmetered without periodic re-blocks;
    only a request dispatched BEFORE the refusal can still return a late 401,
    which tightens once until the immediate revalidation relaxes it;
(b) a 401-triggered revalidation is immediate, not subject to the 60 s
    cadence (only a revalidation whose own answer was a refusal backs off);
(c) a revalidation that returns 200 full scope clears the refusal and normal
    traffic resumes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from solwyn._api_key_errors import Refusal
from solwyn._constants import AUTH_KEY_PATH
from solwyn.client import Solwyn
from solwyn.exceptions import ApiKeyRefusedError
from solwyn.testing import FakeControlPlane

pytestmark = [pytest.mark.unit, pytest.mark.project_validation]

_MESSAGES = [{"role": "user", "content": "hi"}]


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


class _Clock:
    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now


@contextmanager
def _client(
    plane: FakeControlPlane, **options: Any
) -> Iterator[tuple[Solwyn, _OpenAIStub, _Clock]]:
    stub = _OpenAIStub()
    settings: dict[str, Any] = {"lease_enabled": False, "fail_open": True}
    settings.update(options)
    client = plane.wrap(stub, **settings)
    clock = _Clock()
    client._solwyn_binding._clock = clock
    try:
        yield client, stub, clock
    finally:
        client.close()


def _call(client: Solwyn) -> None:
    client.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)


def _effects(client: Solwyn) -> list[str]:
    seen: list[str] = []
    binding = client._solwyn_binding

    def listener() -> None:
        refusal = binding.refusal
        seen.append(refusal.effect if refusal is not None else "cleared")

    binding.add_listener(listener)
    return seen


def _unmetered(plane: FakeControlPlane, client: Solwyn) -> None:
    """Validate, expire the key, and let one call relax to allow_unmetered."""
    client.validate()
    plane.expire_key()
    _call(client)
    refusal = client._solwyn_binding.refusal
    assert refusal is not None and refusal.effect == "allow_unmetered"


def _late_401(client: Solwyn) -> None:
    """A data-plane request dispatched before the refusal answers 401 now."""
    request = httpx.Request("POST", "http://control-plane.invalid/api/v1/budgets/confirm")
    client._solwyn_binding.observe_status(httpx.Response(401, request=request))


def test_a_allow_unmetered_client_making_100_calls_sends_zero_data_plane_requests() -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _client(plane, reporter_flush_interval=0.01) as (client, stub, _clock):
        _unmetered(plane, client)
        reporter = client._solwyn_reporter
        reporter._flush_remaining()
        before = len(plane.project_headers)
        effects = _effects(client)

        for index in range(100):
            _call(client)  # never raises
            if index % 10 == 0:
                reporter._flush_remaining()  # plus the live background uploader
        reporter._flush_remaining()
        traffic = plane.project_headers[before:]

    assert stub.chat.completions.calls == 101
    assert traffic == []  # not one request: no check, no upload, no /auth/key
    assert effects == []  # never re-blocked
    assert reporter.dropped_counts.get("event.api_key_refused", 0) >= 100


def test_a_a_late_401_from_before_the_refusal_blips_once_then_relaxes() -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _client(plane) as (client, stub, _clock):
        _unmetered(plane, client)
        effects = _effects(client)
        auth_before = len(plane.auth_key_requests)

        _late_401(client)
        assert client._solwyn_binding.refusal.effect == "block"
        _call(client)  # the gate revalidates at once and relaxes
        _call(client)

    assert effects == ["block", "allow_unmetered"]
    assert len(plane.auth_key_requests) == auth_before + 1
    assert stub.chat.completions.calls == 3


def test_b_a_401_triggered_revalidation_ignores_the_sixty_second_cadence() -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _client(plane) as (client, stub, clock):
        _unmetered(plane, client)  # the revalidation's own answer was a refusal
        auth_after_relax = len(plane.auth_key_requests)

        clock.now += 1
        _call(client)  # within 60 s of that refusal answer: no revalidation
        assert len(plane.auth_key_requests) == auth_after_relax

        clock.now += 1
        _late_401(client)  # a re-block 2 s after the last revalidation
        _call(client)  # revalidated at once, not after 60 s

    assert len(plane.auth_key_requests) == auth_after_relax + 1
    assert client._solwyn_binding.refusal is not None
    assert client._solwyn_binding.refusal.effect == "allow_unmetered"
    assert stub.chat.completions.calls == 3


def test_b_a_revalidation_answering_a_refusal_backs_off_sixty_seconds() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    with _client(plane) as (client, stub, clock):
        client.validate()
        plane.revoke_key()
        with pytest.raises(ApiKeyRefusedError):
            _call(client)  # 401 -> immediate revalidation -> refused again
        auth_after = len(plane.auth_key_requests)

        for _ in range(3):
            clock.now += 10
            with pytest.raises(ApiKeyRefusedError):
                _call(client)
        assert len(plane.auth_key_requests) == auth_after  # backing off
        clock.now += 31
        with pytest.raises(ApiKeyRefusedError):
            _call(client)

    assert len(plane.auth_key_requests) == auth_after + 1
    assert stub.chat.completions.calls == 0


def test_c_a_200_revalidation_clears_the_refusal_and_traffic_resumes() -> None:
    plane = FakeControlPlane(mode="alert_only")
    with _client(plane) as (client, stub, clock):
        _unmetered(plane, client)
        reporter = client._solwyn_reporter
        reporter._flush_remaining()
        checks_before = len(plane.checks)
        confirms_before = len(plane.confirms)

        plane.restore_key()  # the key works again (for example a rotation)
        clock.now += 61
        _call(client)
        reporter._flush_remaining()

    assert client._solwyn_binding.state == "valid"
    assert client._solwyn_binding.refusal is None
    assert len(plane.checks) == checks_before + 1
    assert len(plane.confirms) == confirms_before + 1
    assert stub.chat.completions.calls == 2


def test_a_blocked_client_drops_queued_items_and_held_leases_without_sending() -> None:
    import solwyn

    plane = FakeControlPlane(mode="hard_deny")
    with _client(plane, lease_enabled=True, reporter_flush_interval=3600) as (
        client,
        _stub,
        _clock,
    ):
        client.validate()
        with solwyn.run("quiet"):
            _call(client)  # holds a lease; its settlement is queued
        _call(client)  # a per-call check; its settlement is queued too
        plane.revoke_key()  # so the immediate revalidation is refused too
        binding = client._solwyn_binding
        binding.refuse(Refusal("api_key_invalid", "hard_deny"), binding.generation)
        before = len(plane.project_headers)
        reporter = client._solwyn_reporter
        reporter._flush_remaining()
        with pytest.raises(ApiKeyRefusedError):
            _call(client)
        budget = client._solwyn_budget
        traffic = [path for path, _ in plane.project_headers[before:]]

    # Only the gate's revalidation (if due) may reach Solwyn.
    assert set(traffic) <= {AUTH_KEY_PATH}
    assert any(key.endswith("api_key_refused") for key in reporter.dropped_counts)
    assert budget._lease.active_run_ids() == []
