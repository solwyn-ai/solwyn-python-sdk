"""Public wrapper helpers and recording lifecycle for ``FakeControlPlane``."""

from __future__ import annotations

import re
from types import SimpleNamespace

import httpx
import pytest

from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import BudgetExceededError
from solwyn.testing import FakeControlPlane


class _SyncCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )


class _SyncChat:
    def __init__(self) -> None:
        self.completions = _SyncCompletions()


class _OpenAIStub:
    def __init__(self) -> None:
        self.chat = _SyncChat()

    def with_options(self, **_kwargs: object) -> _OpenAIStub:
        return self


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


class _AsyncCompletions:
    async def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )


class _AsyncChat:
    def __init__(self) -> None:
        self.completions = _AsyncCompletions()


class _AsyncOpenAIStub:
    def __init__(self) -> None:
        self.chat = _AsyncChat()

    def with_options(self, **_kwargs: object) -> _AsyncOpenAIStub:
        return self


_AsyncOpenAIStub.__module__ = "openai._client"
_AsyncOpenAIStub.__name__ = "AsyncOpenAI"


def _check_payload() -> dict[str, object]:
    return {
        "estimated_input_tokens": 1,
        "model": "gpt-5.5",
        "provider": "openai",
        "fallback_providers": [],
        "fallback_models": [],
        "failover_directive_version": "1",
    }


@pytest.mark.unit
def test_wrap_defaults_to_uncached_checks_and_disables_task_three_leases() -> None:
    plane = FakeControlPlane()

    wrapped = plane.wrap(_OpenAIStub())

    assert isinstance(wrapped, Solwyn)
    assert wrapped._config.api_key == plane.api_key
    assert wrapped._config.api_url == plane.api_url
    assert wrapped._config.budget_check_cache_ttl == 0
    assert wrapped._config.lease_enabled is False
    wrapped.close()


@pytest.mark.unit
def test_wrap_allows_callers_to_override_wrapper_defaults() -> None:
    plane = FakeControlPlane()

    wrapped = plane.wrap(
        _OpenAIStub(),
        budget_check_cache_ttl=9,
        lease_enabled=True,
    )

    assert wrapped._config.budget_check_cache_ttl == 9
    assert wrapped._config.lease_enabled is True
    wrapped.close()


@pytest.mark.unit
async def test_wrap_async_uses_the_dual_transport_and_same_defaults() -> None:
    plane = FakeControlPlane()

    wrapped = plane.wrap_async(_AsyncOpenAIStub())

    assert isinstance(wrapped, AsyncSolwyn)
    assert wrapped._config.api_url == plane.api_url
    assert wrapped._config.budget_check_cache_ttl == 0
    assert wrapped._config.lease_enabled is False
    await wrapped.close()


@pytest.mark.unit
def test_magic_deny_blocks_real_wrapper_before_provider_dispatch() -> None:
    plane = FakeControlPlane()
    provider = _OpenAIStub()
    wrapped = plane.wrap(provider)

    with pytest.raises(BudgetExceededError) as captured:
        wrapped.chat.completions.create(
            model="solwyn-test/deny",
            messages=[],
        )
    wrapped.close()

    assert captured.value.budget_period == "monthly"
    assert provider.chat.completions.calls == 0
    assert plane.checks[0].model == "solwyn-test/deny"


@pytest.mark.unit
def test_api_key_is_deterministic_and_valid_project_key_shape() -> None:
    first = FakeControlPlane()
    second = FakeControlPlane()

    assert first.api_key == second.api_key
    assert re.fullmatch(r"sk_proj_[0-9a-f]{64}", first.api_key)
    assert first.api_url == "http://control-plane.invalid"


@pytest.mark.unit
def test_reset_recording_clears_only_recorded_requests() -> None:
    plane = FakeControlPlane()
    plane.deny_next(2)

    with httpx.Client(transport=plane.transport) as client:
        first = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(),
        )
        client.get(f"{plane.api_url}/unknown")
        plane.reset_recording()
        second = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(),
        )

    assert first.json()["allowed"] is False
    assert second.json()["allowed"] is False
    assert len(plane.checks) == 1
    assert plane.unmatched_requests == []
    assert plane.confirms == []
    assert plane.ingested == []
    assert plane.untracked_reports == []
    assert plane.breaker_reports == []
