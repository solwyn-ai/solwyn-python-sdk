"""Tests that event builders preserve captured run metadata."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from conftest import VALID_API_KEY, call_uuid

import solwyn
from solwyn._base import _SolwynBase
from solwyn._registry import build_runtimes
from solwyn._run import _capture_run_context
from solwyn._types import CallStatus, MetadataEvent, ProviderEntry, ProviderName
from solwyn.config import SolwynConfig


def _make_base() -> _SolwynBase:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    runtimes = build_runtimes(client, "gpt-5.5", [])
    config = SolwynConfig(
        api_key=VALID_API_KEY,
        providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
    )
    return _SolwynBase(config, runtimes)


def _build(base: _SolwynBase) -> MetadataEvent:
    return base._build_metadata_event(
        model="gpt-5.5",
        provider="openai",
        input_tokens=10,
        output_tokens=5,
        token_details=None,
        latency_ms=12.3,
        status=CallStatus.SUCCESS,
        is_model_fallback=False,
        call_id=call_uuid("call_run_emit"),
    )


@pytest.mark.unit
class TestEmitWithActiveRun:
    """Inside a run scope, the wire fields must be populated."""

    def test_outside_scope_fields_are_none(self) -> None:
        base = _make_base()
        event = _build(base)
        assert event.agent_run_id is None
        assert event.agent_run_name is None
        assert event.tags is None

    def test_inside_scope_fields_are_set(self) -> None:
        base = _make_base()
        with solwyn.run("nightly-batch", tags={"team": "research"}) as run_id:
            event = _build(base)
        assert event.agent_run_id == run_id
        assert event.agent_run_name == "nightly-batch"
        assert event.tags == {"team": "research"}

    def test_after_scope_fields_revert_to_none(self) -> None:
        base = _make_base()
        with solwyn.run("nightly-batch"):
            pass
        event = _build(base)
        assert event.agent_run_id is None
        assert event.agent_run_name is None
        assert event.tags is None

    def test_error_event_also_tagged(self) -> None:
        base = _make_base()
        with solwyn.run("nightly-batch", tags={"team": "research"}) as run_id:
            event = base._build_error_event(
                model="gpt-5.5",
                provider="openai",
                latency_ms=12.3,
                is_model_fallback=False,
                call_id=call_uuid("call_run_emit_error"),
            )
        assert event.agent_run_id == run_id
        assert event.agent_run_name == "nightly-batch"
        assert event.tags == {"team": "research"}

    def test_explicit_snapshot_survives_scope_exit(self) -> None:
        base = _make_base()
        with solwyn.run("nightly-batch", tags={"team": "research"}) as run_id:
            snapshot = _capture_run_context()

        event = base._build_metadata_event(
            model="gpt-5.5",
            provider="openai",
            input_tokens=10,
            output_tokens=5,
            token_details=None,
            latency_ms=12.3,
            status=CallStatus.SUCCESS,
            is_model_fallback=False,
            call_id=call_uuid("call_run_snapshot"),
            agent_run=snapshot,
        )
        assert event.agent_run_id == run_id
        assert event.agent_run_name == "nightly-batch"
        assert event.tags == {"team": "research"}

    @pytest.mark.asyncio
    async def test_async_concurrent_tasks_tag_independently(self) -> None:
        base = _make_base()

        async def emit_under(name: str) -> tuple[str, str | None, str | None]:
            async with solwyn.run(name) as run_id:
                await asyncio.sleep(0)
                event = _build(base)
                return run_id, event.agent_run_id, event.agent_run_name

        a, b = await asyncio.gather(emit_under("task-a"), emit_under("task-b"))
        # Each task's event carries its own run id, never the sibling's.
        assert a[0] == a[1] and a[2] == "task-a"
        assert b[0] == b[1] and b[2] == "task-b"
        assert a[0] != b[0]
