"""End-to-end request-scoped price-hint acceptance tests.

The provider SDK boundaries are duck-typed fakes; every Solwyn component from
the public wrapper through the FakeControlPlane wire models stays real.  Each
test asserts the control-plane recordings after shutdown, rather than calls
made to a provider fake.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from solwyn import _base
from solwyn._routing import CostPolicy
from solwyn._types import CallStatus, FailoverReason
from solwyn.testing import FakeControlPlane

_COST_POLICY_WARNING = (
    "CostPolicy selected but this budget check carried no price hints; using health-based order"
)
_CHAT_REQUEST = {"messages": [{"role": "user", "content": "hi"}]}


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    client.chat.completions.create.return_value = _openai_response()
    client.embeddings.create.return_value = _embedding_response()
    return client


def _anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"
    client.with_options.return_value = client
    client.messages.create.return_value = _anthropic_response()
    return client


def _async_openai_client() -> MagicMock:
    client = _openai_client()
    client.chat.completions.create = AsyncMock(return_value=_openai_response())
    return client


def _async_anthropic_client() -> MagicMock:
    client = _anthropic_client()
    client.messages.create = AsyncMock(return_value=_anthropic_response())
    return client


def _openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content="ok", tool_calls=None),
                finish_reason="stop",
                index=0,
            )
        ],
        model="gpt-5.4-nano",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        model="claude-haiku-4-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _embedding_response() -> SimpleNamespace:
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10))


def _hinted_providers(hints: dict[str, float]) -> list[str]:
    """Derive expected order directly from the API-shaped relative-ratio map."""
    return [provider for provider, _ in sorted(hints.items(), key=lambda item: item[1])]


@pytest.fixture
def reset_cost_policy_warning() -> Iterator[None]:
    """Isolate CostPolicy's process-wide absent-hints warning latch."""
    _base._cost_policy_inactive_warned = False
    yield
    _base._cost_policy_inactive_warned = False


@pytest.mark.unit
def test_second_call_with_different_primary_model_is_never_ordered_by_first_calls_hints() -> None:
    """Catches a cache key that omits the requested primary model and replays stale hints."""
    plane = FakeControlPlane(price_hints={"openai": 3.0, "anthropic": 1.0})
    openai, anthropic = _openai_client(), _anthropic_client()
    first_hints = dict(plane.price_hints)
    client = plane.wrap(
        openai,
        model="gpt-5.4-nano",
        fallback=[(anthropic, "claude-haiku-4-5", {"max_tokens": 64})],
        selection_policy=CostPolicy(),
        lease_enabled=False,
        budget_check_cache_ttl=5,
    )
    try:
        client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
        plane.price_hints = {"openai": 1.0, "anthropic": 5.0}
        second_hints = dict(plane.price_hints)
        client.chat.completions.create(model="gpt-5.5-pro", **_CHAT_REQUEST)
    finally:
        client.close()

    assert [check.model for check in plane.checks] == ["gpt-5.4-nano", "gpt-5.5-pro"]
    assert all(check.price_hints_version == "1" for check in plane.checks)
    served = [event for event in plane.ingested if event.status is CallStatus.SUCCESS]
    assert [event.provider.value for event in served] == [
        _hinted_providers(first_hints)[0],
        _hinted_providers(second_hints)[0],
    ]
    assert [event.failover_reason for event in served] == [FailoverReason.COST_ROUTED, None]


@pytest.mark.unit
def test_same_chain_within_ttl_hits_cache_and_replays_its_own_hints() -> None:
    """Catches cached checks that drop the original hint map before rerouting."""
    plane = FakeControlPlane(price_hints={"openai": 3.0, "anthropic": 1.0})
    openai, anthropic = _openai_client(), _anthropic_client()
    served_hints = dict(plane.price_hints)
    client = plane.wrap(
        openai,
        model="gpt-5.4-nano",
        fallback=[(anthropic, "claude-haiku-4-5", {"max_tokens": 64})],
        selection_policy=CostPolicy(),
        lease_enabled=False,
        budget_check_cache_ttl=5,
    )
    try:
        client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
        client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
    finally:
        client.close()

    assert len(plane.checks) == 1
    served = [event for event in plane.ingested if event.status is CallStatus.SUCCESS]
    assert [event.provider.value for event in served] == [_hinted_providers(served_hints)[0]] * 2
    assert [event.failover_reason for event in served] == [
        FailoverReason.COST_ROUTED,
        FailoverReason.COST_ROUTED,
    ]


@pytest.mark.unit
def test_modality_change_within_ttl_rechecks() -> None:
    """Catches a cache key that treats chat and embedding preflights as interchangeable."""
    plane = FakeControlPlane()
    client = plane.wrap(
        _openai_client(),
        model="gpt-5.4-nano",
        lease_enabled=False,
        budget_check_cache_ttl=5,
    )
    try:
        client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
        client.embeddings.create(model="text-embedding-3-small", input=["hi"])
    finally:
        client.close()

    assert len(plane.checks) == 2
    assert plane.checks[1].modality == "embedding"


@pytest.mark.unit
def test_empty_hints_restore_configured_order_without_warning(
    reset_cost_policy_warning: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches an explicit server hint clear being treated as stale cost-routing input."""
    plane = FakeControlPlane(price_hints={})
    client = plane.wrap(
        _openai_client(),
        model="gpt-5.4-nano",
        fallback=[(_anthropic_client(), "claude-haiku-4-5", {"max_tokens": 64})],
        selection_policy=CostPolicy(),
        lease_enabled=False,
        budget_check_cache_ttl=5,
    )
    try:
        with caplog.at_level(logging.WARNING):
            client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
    finally:
        client.close()

    served = [event for event in plane.ingested if event.status is CallStatus.SUCCESS]
    assert [event.provider.value for event in served] == ["openai"]
    assert [event.failover_reason for event in served] == [None]
    assert _COST_POLICY_WARNING not in caplog.text


@pytest.mark.unit
def test_null_hints_restore_configured_order_and_warn_once(
    reset_cost_policy_warning: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Catches absent-hint fallback that either reorders traffic or logs once per call."""
    plane = FakeControlPlane()
    client = plane.wrap(
        _openai_client(),
        model="gpt-5.4-nano",
        fallback=[(_anthropic_client(), "claude-haiku-4-5", {"max_tokens": 64})],
        selection_policy=CostPolicy(),
        lease_enabled=False,
        budget_check_cache_ttl=5,
    )
    try:
        with caplog.at_level(logging.WARNING):
            client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
            client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
    finally:
        client.close()

    served = [event for event in plane.ingested if event.status is CallStatus.SUCCESS]
    assert [event.provider.value for event in served] == ["openai", "openai"]
    assert [event.failover_reason for event in served] == [None, None]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if _COST_POLICY_WARNING in record.getMessage()
    ]
    assert warnings == [_COST_POLICY_WARNING]


@pytest.mark.unit
def test_health_policy_ignores_hints_entirely() -> None:
    """Catches the default health policy accidentally consuming CostPolicy's server signal."""
    plane = FakeControlPlane(price_hints={"openai": 3.0, "anthropic": 1.0})
    client = plane.wrap(
        _openai_client(),
        model="gpt-5.4-nano",
        fallback=[(_anthropic_client(), "claude-haiku-4-5", {"max_tokens": 64})],
        lease_enabled=False,
        budget_check_cache_ttl=5,
    )
    try:
        client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
        client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
    finally:
        client.close()

    assert len(plane.checks) == 1
    served = [event for event in plane.ingested if event.status is CallStatus.SUCCESS]
    assert [event.provider.value for event in served] == ["openai", "openai"]
    assert [event.failover_reason for event in served] == [None, None]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_second_call_with_different_primary_model_is_never_ordered_by_first_calls_hints(
) -> None:
    """Catches the async cache key omitting the requested model and replaying stale hints."""
    plane = FakeControlPlane(price_hints={"openai": 3.0, "anthropic": 1.0})
    first_hints = dict(plane.price_hints)
    async with plane.wrap_async(
        _async_openai_client(),
        model="gpt-5.4-nano",
        fallback=[(_async_anthropic_client(), "claude-haiku-4-5", {"max_tokens": 64})],
        selection_policy=CostPolicy(),
        lease_enabled=False,
        budget_check_cache_ttl=5,
    ) as client:
        await client.chat.completions.create(model="gpt-5.4-nano", **_CHAT_REQUEST)
        plane.price_hints = {"openai": 1.0, "anthropic": 5.0}
        second_hints = dict(plane.price_hints)
        await client.chat.completions.create(model="gpt-5.5-pro", **_CHAT_REQUEST)

    assert [check.model for check in plane.checks] == ["gpt-5.4-nano", "gpt-5.5-pro"]
    assert all(check.price_hints_version == "1" for check in plane.checks)
    served = [event for event in plane.ingested if event.status is CallStatus.SUCCESS]
    assert [event.provider.value for event in served] == [
        _hinted_providers(first_hints)[0],
        _hinted_providers(second_hints)[0],
    ]
    assert [event.failover_reason for event in served] == [FailoverReason.COST_ROUTED, None]
