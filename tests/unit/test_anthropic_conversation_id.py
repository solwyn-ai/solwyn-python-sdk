"""Conversation options follow the attempted client's mapping, never request kwargs."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx
import pytest

import solwyn
from solwyn._base import MediaSurfaceSpec
from solwyn._token_details import TokenDetails
from solwyn.testing import FakeControlPlane

_MISSING = object()


def _create(**kwargs: object) -> object:
    raise NotImplementedError


def _with_options(**kwargs: object) -> object:
    raise NotImplementedError


def _client(
    is_async: bool,
    option: object = _MISSING,
    *,
    descriptor: object = _MISSING,
    dialect: str = "anthropic",
    class_option: bool = False,
) -> Any:
    """A declared provider shape with real attributes and mocked I/O only."""
    name = "Anthropic" if dialect == "anthropic" else "OpenAI"
    attributes = {"__module__": f"{dialect}._client"}
    if descriptor is not _MISSING:
        attributes["open_telemetry"] = descriptor
    elif class_option:
        attributes["open_telemetry"] = option
    client = type(f"Async{name}" if is_async else name, (), attributes)()
    if option is not _MISSING and not class_option:
        client.open_telemetry = option
    client.with_options = MagicMock(spec=_with_options, return_value=client)
    response = SimpleNamespace(
        id="msg_test",
        role="assistant",
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        stop_sequence=None,
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    create = (AsyncMock if is_async else MagicMock)(spec=_create, return_value=response)
    client.messages = SimpleNamespace(create=create)
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return client


def _wrap(client: Any, is_async: bool, **kwargs: Any) -> Any:
    plane = FakeControlPlane()
    wrap = plane.wrap_async if is_async else plane.wrap
    return wrap(client, lease_enabled=False, **kwargs)


async def _await_if_needed(result: Any) -> Any:
    return await result if inspect.isawaitable(result) else result


async def _call(wrapper: Any, *, stream: bool = False) -> Any:
    return await _await_if_needed(
        wrapper._intercepted_call(
            model=wrapper._solwyn_runtimes[0].entry.model or "claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            stream=stream,
        )
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("source", ["instance", "class", "property", "empty"])
async def test_extends_existing_mapping_without_mutating_customer_options(
    is_async: bool, source: str
) -> None:
    nested = MappingProxyType({"enabled": False})
    original = (
        {}
        if source == "empty"
        else {
            "tracer_provider": object(),
            "propagation": False,
            "traces": nested,
        }
    )
    mapping = MappingProxyType(original)
    getter_calls: list[object] = []

    def getter(client: object) -> object:
        getter_calls.append(client)
        return mapping

    client = (
        _client(is_async, descriptor=property(getter))
        if source == "property"
        else _client(is_async, mapping, class_option=source == "class")
    )
    wrapper = _wrap(client, is_async, anthropic_conversation_id=True)
    try:
        with solwyn.run("run name is not a conversation id", tags={"private": "tag"}) as run_id:
            await _call(wrapper)

        options = client.with_options.call_args.kwargs
        assert options["max_retries"] == 0
        extended = options["open_telemetry"]
        assert extended == {**mapping, "conversation_id": run_id}
        assert extended is not mapping
        assert "conversation_id" not in original
        if source != "empty":
            assert extended["tracer_provider"] is original["tracer_provider"]
            assert extended["traces"] is nested
            assert extended["propagation"] is False
        assert getter_calls == ([client] if source == "property" else [])
        sent = client.messages.create.call_args.kwargs
        assert "conversation_id" not in sent
        assert "open_telemetry" not in sent
    finally:
        await _await_if_needed(wrapper.close())


class _Descriptor:
    calls = 0

    def __get__(self, instance: object, owner: type | None = None) -> object:
        self.calls += 1
        raise RuntimeError("must not evaluate this descriptor")


class _RaisingMapping(Mapping[str, object]):
    def __init__(self, failure: str) -> None:
        self.failure = failure

    def __contains__(self, key: object) -> bool:
        if self.failure == "contains":
            raise RuntimeError("unreadable mapping membership")
        return False

    def __iter__(self) -> Iterator[str]:
        if self.failure == "iter":
            raise RuntimeError("unreadable mapping keys")
        return iter(("tracer_provider",))

    def __getitem__(self, key: str) -> object:
        if self.failure == "getitem":
            raise RuntimeError("unreadable mapping value")
        return None

    def __len__(self) -> int:
        return 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("surface", ["chat", "media"])
@pytest.mark.parametrize("failure", ["contains", "iter", "getitem"])
async def test_unreadable_mapping_does_not_interrupt_provider_dispatch(
    is_async: bool, surface: str, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(is_async, _RaisingMapping(failure))
    wrapper = _wrap(client, is_async, anthropic_conversation_id=True)

    def prepare_media_call(self: object, surface: str, client: Any, kwargs: Any, **_: Any) -> Any:
        return client.messages.create, kwargs

    if surface == "media":
        monkeypatch.setattr(
            type(wrapper._solwyn_runtimes[0].adapter),
            "prepare_media_call",
            prepare_media_call,
            raising=False,
        )
    try:
        with solwyn.run("test"):
            if surface == "media":
                spec = MediaSurfaceSpec(
                    surface="embeddings",
                    modality="embedding",
                    extract_usage=lambda _: TokenDetails(input_tokens=10),
                    measure_request=lambda _: None,
                )
                response = await _await_if_needed(
                    wrapper._media_call(spec, model="claude-sonnet-5")
                )
            else:
                response = await _call(wrapper)
        assert response.usage.input_tokens == 10
        assert client.with_options.call_count == 1
        assert "open_telemetry" not in client.with_options.call_args.kwargs
        client.messages.create.assert_called_once()
    finally:
        await _await_if_needed(wrapper.close())


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "condition",
    [
        "disabled",
        "no-run",
        "missing",
        "none",
        "false",
        "true",
        "list",
        "string",
        "existing",
        "existing-none",
        "existing-empty",
        "other-dialect",
        "descriptor",
        "getter-raises",
        "write-only-property",
    ],
)
async def test_skip_conditions_leave_client_options_alone(is_async: bool, condition: str) -> None:
    getter_calls: list[object] = []

    def getter(client: object) -> object:
        getter_calls.append(client)
        if condition == "getter-raises":
            raise RuntimeError("getter failure must not break provider dispatch")
        return {}

    descriptor = _Descriptor()
    values = {
        "missing": _MISSING,
        "none": None,
        "false": False,
        "true": True,
        "list": [],
        "string": "enabled",
        "existing": {"conversation_id": "customer-id"},
        "existing-none": {"conversation_id": None},
        "existing-empty": {"conversation_id": ""},
    }
    if condition == "descriptor":
        client = _client(is_async, descriptor=descriptor)
    elif condition == "write-only-property":
        client = _client(is_async, descriptor=property())
    elif condition in {"disabled", "no-run", "getter-raises", "other-dialect"}:
        client = _client(
            is_async,
            descriptor=property(getter),
            dialect="openai" if condition == "other-dialect" else "anthropic",
        )
    else:
        client = _client(is_async, values[condition])
    wrapper = _wrap(client, is_async, anthropic_conversation_id=condition != "disabled")
    try:
        with nullcontext() if condition == "no-run" else solwyn.run("test"):
            await _call(wrapper)

        assert "open_telemetry" not in client.with_options.call_args.kwargs
        assert descriptor.calls == 0
        assert len(getter_calls) == (1 if condition == "getter-raises" else 0)
        client.messages.create.assert_called_once()
    finally:
        await _await_if_needed(wrapper.close())


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_magicmock_synthesized_attribute_is_skipped(
    is_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(is_async)
    wrapper = _wrap(client, is_async, anthropic_conversation_id=True)
    mock = MagicMock(spec=["open_telemetry"])
    getter_calls: list[str] = []

    def synthesize(self: object, name: str) -> object:
        getter_calls.append(name)
        return {}

    # A MagicMock __getattr__ that synthesizes a mapping must never be called.
    monkeypatch.setattr(type(mock), "__getattr__", synthesize)
    runtime = wrapper._solwyn_runtimes[0]

    try:
        with solwyn.run("test") as run_id:
            options = wrapper._extra_client_options(
                replace(runtime, sdk_client=mock), run_id=run_id
            )
        assert options == {}
        assert "open_telemetry" not in vars(mock)
        assert getter_calls == []
    finally:
        await _await_if_needed(wrapper.close())


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("primary_dialect", ["anthropic", "openai"])
async def test_failover_uses_each_anthropic_clients_own_mapping(
    is_async: bool, primary_dialect: str
) -> None:
    primary_options = {"propagation": True, "tracer_provider": object()}
    fallback_options = {"propagation": False, "tracer_provider": object()}
    primary = _client(is_async, primary_options, dialect=primary_dialect)
    fallback = _client(is_async, descriptor=property(lambda _: fallback_options))
    primary.messages.create.side_effect = httpx.ConnectError("connect failed")
    wrapper = _wrap(
        primary,
        is_async,
        anthropic_conversation_id=True,
        fallback=[(fallback, "claude-sonnet-5")],
    )
    try:
        with solwyn.run("failover") as run_id:
            await _call(wrapper)
        assert fallback.with_options.call_args.kwargs["open_telemetry"] == {
            **fallback_options,
            "conversation_id": run_id,
        }
        if primary_dialect == "anthropic":
            assert primary.with_options.call_args.kwargs["open_telemetry"] == {
                **primary_options,
                "conversation_id": run_id,
            }
        else:
            assert "open_telemetry" not in primary.with_options.call_args.kwargs
        assert "conversation_id" not in primary_options
        assert "conversation_id" not in fallback_options
    finally:
        await _await_if_needed(wrapper.close())


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_nested_and_detached_runs_do_not_leak_options(is_async: bool) -> None:
    client = _client(is_async, {})
    wrapper = _wrap(client, is_async, anthropic_conversation_id=True, velocity_mode="off")
    handle = solwyn.create_run("detached")
    try:
        with solwyn.run("parent") as parent:
            await _call(wrapper)
            with solwyn.run("child") as child:
                await _call(wrapper)
            await _call(wrapper)
        with handle.activate():
            await _call(wrapper)
        await _call(wrapper)
        options = [call.kwargs.get("open_telemetry") for call in client.with_options.call_args_list]
        assert options == [
            {"conversation_id": parent},
            {"conversation_id": child},
            {"conversation_id": parent},
            {"conversation_id": handle.run_id},
            None,
        ]
        for run_id in (parent, child, handle.run_id):
            assert run_id.startswith("run_")
            assert UUID(run_id.removeprefix("run_")).version == 4
        assert client.open_telemetry == {}
    finally:
        handle.finish()
        await _await_if_needed(wrapper.close())


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_media_dispatch_receives_extra_client_options(
    is_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(is_async, {})
    wrapper = _wrap(client, is_async, anthropic_conversation_id=True)

    def prepare_media_call(self: object, surface: str, client: Any, kwargs: Any, **_: Any) -> Any:
        return client.messages.create, kwargs

    monkeypatch.setattr(
        type(wrapper._solwyn_runtimes[0].adapter),
        "prepare_media_call",
        prepare_media_call,
        raising=False,
    )
    spec = MediaSurfaceSpec(
        surface="embeddings",
        modality="embedding",
        extract_usage=lambda _: TokenDetails(input_tokens=10),
        measure_request=lambda _: None,
    )
    try:
        with solwyn.run("media") as run_id:
            await _await_if_needed(wrapper._media_call(spec, model="claude-sonnet-5"))
        assert client.with_options.call_args.kwargs["open_telemetry"] == {
            "conversation_id": run_id,
        }
    finally:
        await _await_if_needed(wrapper.close())
