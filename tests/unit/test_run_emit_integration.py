"""Tests that ``_build_metadata_event`` tags events with the active run.

The contextvar set by ``solwyn.run("...")`` is read at event-build time —
no parameter is threaded through the emit path. This file verifies that
seam without exercising the full HTTP layer.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from conftest import VALID_API_KEY

import solwyn
from solwyn._base import _SolwynBase
from solwyn._registry import build_runtimes
from solwyn._types import CallStatus, MetadataEvent, ProviderEntry, ProviderName
from solwyn.config import SolwynConfig


def _make_base() -> _SolwynBase:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    runtimes = build_runtimes(client, "gpt-4o", [])
    config = SolwynConfig(
        api_key=VALID_API_KEY,
        providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-4o")],
    )
    return _SolwynBase(config, runtimes)


def _build(base: _SolwynBase) -> MetadataEvent:
    return base._build_metadata_event(
        model="gpt-4o",
        provider="openai",
        input_tokens=10,
        output_tokens=5,
        token_details=None,
        latency_ms=12.3,
        status=CallStatus.SUCCESS,
        is_model_fallback=False,
    )


@pytest.mark.unit
class TestEmitWithActiveRun:
    """Inside a run scope, the wire fields must be populated."""

    def test_outside_scope_fields_are_none(self) -> None:
        base = _make_base()
        event = _build(base)
        assert event.agent_run_id is None
        assert event.agent_run_name is None

    def test_inside_scope_fields_are_set(self) -> None:
        base = _make_base()
        with solwyn.run("nightly-batch") as run_id:
            event = _build(base)
        assert event.agent_run_id == run_id
        assert event.agent_run_name == "nightly-batch"

    def test_after_scope_fields_revert_to_none(self) -> None:
        base = _make_base()
        with solwyn.run("nightly-batch"):
            pass
        event = _build(base)
        assert event.agent_run_id is None
        assert event.agent_run_name is None

    def test_error_event_also_tagged(self) -> None:
        base = _make_base()
        with solwyn.run("nightly-batch") as run_id:
            event = base._build_error_event(
                model="gpt-4o",
                provider="openai",
                latency_ms=12.3,
                is_model_fallback=False,
            )
        assert event.agent_run_id == run_id
        assert event.agent_run_name == "nightly-batch"

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
