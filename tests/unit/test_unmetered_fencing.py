"""Unmetered surfaces share the metered gate and stay fenced (review round 2).

* Unmetered entry paths run the same single-flight revalidation as the metered
  gate: after a reporter 401 on an alert_only project, unmetered calls
  re-check at once and relax to ``allow_unmetered`` instead of blocking.
* A derived client (``with_options``) keeps its context-manager protocol, and
  the value it enters stays fenced.
* Defence in depth on unmetered spend handles: a realtime manager's public
  ``enter()``, pages produced by raw or streaming response envelopes, and
  every page an ``iter_pages()`` iterator fetches all honour a refusal.
"""

from __future__ import annotations

import inspect
import threading
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import httpx
import openai
import pytest

from solwyn._api_key_errors import Refusal
from solwyn._types import MetadataEvent
from solwyn.exceptions import ApiKeyRefusedError
from solwyn.testing import FakeControlPlane

pytestmark = [pytest.mark.unit, pytest.mark.project_validation]

_BATCH: dict[str, Any] = {
    "input_file_id": "file_1",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h",
}


class _Wire:
    """The provider: every list answers a page that has a next page."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "batch_1", "object": "batch"}],
                    "has_more": True,
                    "first_id": "batch_1",
                    "last_id": "batch_1",
                },
            )
        return httpx.Response(200, json={"id": "batch_1", "object": "batch"})

    async def ahandle(self, request: httpx.Request) -> httpx.Response:
        return self.handle(request)


def _sync_raw(wire: _Wire) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(wire.handle)),
    )


def _async_raw(wire: _Wire) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="provider-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire.ahandle)),
    )


def _block(wrapped: Any) -> None:
    binding = wrapped._solwyn_binding
    binding.refuse(Refusal("read_only_key"), binding.generation)


def _metadata_event() -> MetadataEvent:
    return MetadataEvent(
        model="gpt-5.5",
        provider="openai",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        status="success",
        is_model_fallback=False,
        call_id=str(uuid4()),
        sdk_instance_id=str(uuid4()),
        timestamp=datetime.now(UTC),
    )


# ── unmetered entry paths run the gate's revalidation ──────────────────


def test_sync_unmetered_calls_revalidate_after_a_reporter_401() -> None:
    plane = FakeControlPlane(mode="alert_only")
    wire = _Wire()
    wrapped = plane.wrap(
        _sync_raw(wire),
        on_unmetered="allow",
        report_untracked_surfaces=False,
        reporter_flush_interval=3600,
    )
    try:
        wrapped.validate()
        plane.expire_key()
        reporter = threading.Thread(
            target=lambda: wrapped._solwyn_reporter._send_batch([_metadata_event()])
        )
        reporter.start()
        reporter.join(5)
        binding = wrapped._solwyn_binding
        assert binding.refusal is not None and binding.refusal.effect == "block"
        assert binding.revalidation_due()
        auth_before = len(plane.auth_key_requests)

        for _ in range(100):
            wrapped.batches.create(**_BATCH)  # relaxes instead of raising

        assert len(plane.auth_key_requests) == auth_before + 1
        assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"
        assert len(wire.paths) == 100
    finally:
        wrapped.close()


async def test_async_unmetered_calls_revalidate_after_a_reporter_401() -> None:
    plane = FakeControlPlane(mode="alert_only")
    wire = _Wire()
    wrapped = plane.wrap_async(
        _async_raw(wire),
        on_unmetered="allow",
        report_untracked_surfaces=False,
        reporter_flush_interval=3600,
    )
    try:
        await wrapped.validate()
        plane.expire_key()
        await wrapped._solwyn_reporter._send_batch([_metadata_event()])
        binding = wrapped._solwyn_binding
        assert binding.refusal is not None and binding.refusal.effect == "block"
        assert binding.revalidation_due()
        auth_before = len(plane.auth_key_requests)

        for _ in range(100):
            await wrapped.batches.create(**_BATCH)

        assert len(plane.auth_key_requests) == auth_before + 1
        assert binding.refusal is not None and binding.refusal.effect == "allow_unmetered"
        assert len(wire.paths) == 100
    finally:
        await wrapped.close()


def test_an_unmetered_call_honours_the_sixty_second_backoff_on_hard_deny() -> None:
    plane = FakeControlPlane(mode="hard_deny")
    wire = _Wire()
    wrapped = plane.wrap(_sync_raw(wire), on_unmetered="allow", report_untracked_surfaces=False)
    try:
        wrapped.validate()
        plane.revoke_key()
        binding = wrapped._solwyn_binding
        request = httpx.Request("POST", "http://control-plane.invalid/api/v1/budgets/confirm")
        binding.observe_status(httpx.Response(401, request=request))
        auth_before = len(plane.auth_key_requests)

        for _ in range(20):
            with pytest.raises(ApiKeyRefusedError):
                wrapped.batches.create(**_BATCH)

        assert len(plane.auth_key_requests) == auth_before + 1  # then backs off 60 s
        assert wire.paths == []
    finally:
        wrapped.close()


# ── derived clients keep their context-manager protocol ────────────────


def test_a_derived_sync_client_is_still_a_fenced_context_manager() -> None:
    wire = _Wire()
    wrapped = FakeControlPlane().wrap(_sync_raw(wire), on_unmetered="allow")
    try:
        with wrapped.with_options(timeout=5) as derived:
            assert isinstance(derived, openai.OpenAI)
            derived.batches.create(**_BATCH)
            assert len(wire.paths) == 1
            _block(wrapped)
            with pytest.raises(ApiKeyRefusedError):
                derived.batches.create(**_BATCH)
        assert len(wire.paths) == 1
    finally:
        wrapped.close()


async def test_a_derived_async_client_is_still_a_fenced_context_manager() -> None:
    wire = _Wire()
    wrapped = FakeControlPlane().wrap_async(_async_raw(wire), on_unmetered="allow")
    try:
        async with wrapped.with_options(timeout=5) as derived:
            assert isinstance(derived, openai.AsyncOpenAI)
            await derived.batches.create(**_BATCH)
            assert len(wire.paths) == 1
            _block(wrapped)
            with pytest.raises(ApiKeyRefusedError):
                await derived.batches.create(**_BATCH)
        assert len(wire.paths) == 1
    finally:
        await wrapped.close()


def test_a_fenced_page_is_not_mistaken_for_a_context_manager() -> None:
    wire = _Wire()
    wrapped = FakeControlPlane().wrap(_sync_raw(wire), on_unmetered="allow")
    try:
        page = wrapped.batches.list()
        assert not hasattr(page, "__enter__")
        assert not hasattr(page, "__aenter__")
    finally:
        wrapped.close()


# ── realtime manager public enter() ───────────────────────────────────


class _Socket:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    def send(self, event: Any, *_args: object, **_kwargs: object) -> None:
        self.sent.append(event)

    def close(self, *_args: object, **_kwargs: object) -> None:
        pass


class _AsyncSocket(_Socket):
    async def send(self, event: Any, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        self.sent.append(event)

    async def close(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        pass


@pytest.mark.parametrize("enter_before_refusal", [False, True])
def test_a_realtime_managers_public_enter_is_fenced(enter_before_refusal: bool) -> None:
    from openai.resources.realtime.realtime import RealtimeConnectionManager

    socket = _Socket()
    wrapped = FakeControlPlane().wrap(_sync_raw(_Wire()), on_unmetered="allow")
    try:
        manager = wrapped.realtime.connect(model="gpt-realtime")
        with patch.object(RealtimeConnectionManager, "_connect_ws", return_value=socket) as ws:
            if enter_before_refusal:
                connection = manager.enter()
            _block(wrapped)
            opened = ws.call_count
            with pytest.raises(ApiKeyRefusedError):
                if not enter_before_refusal:
                    connection = manager.enter()
                connection.response.create()
            assert ws.call_count == opened
            assert socket.sent == []
    finally:
        wrapped.close()


@pytest.mark.parametrize("enter_before_refusal", [False, True])
async def test_an_async_realtime_managers_public_enter_is_fenced(
    enter_before_refusal: bool,
) -> None:
    from openai.resources.realtime.realtime import AsyncRealtimeConnectionManager

    socket = _AsyncSocket()
    wrapped = FakeControlPlane().wrap_async(_async_raw(_Wire()), on_unmetered="allow")
    try:
        manager = wrapped.realtime.connect(model="gpt-realtime")
        with patch.object(AsyncRealtimeConnectionManager, "_connect_ws", return_value=socket) as ws:
            if enter_before_refusal:
                connection = await manager.enter()
            _block(wrapped)
            opened = ws.call_count
            with pytest.raises(ApiKeyRefusedError):
                if not enter_before_refusal:
                    connection = await manager.enter()
                await connection.response.create()
            assert ws.call_count == opened
            assert socket.sent == []
    finally:
        await wrapped.close()


# ── pages produced by response envelopes ───────────────────────────────


@pytest.mark.parametrize("kind", ["raw", "streaming"])
@pytest.mark.parametrize("parse_after_refusal", [False, True])
def test_pages_parsed_from_sync_envelopes_stay_fenced(kind: str, parse_after_refusal: bool) -> None:
    wire = _Wire()
    wrapped = FakeControlPlane().wrap(_sync_raw(wire), on_unmetered="allow")
    manager = None
    try:
        if kind == "raw":
            response = wrapped.batches.with_raw_response.list()
        else:
            manager = wrapped.batches.with_streaming_response.list()
            response = manager.__enter__()
        if not parse_after_refusal:
            page = response.parse()
        _block(wrapped)
        sent = len(wire.paths)
        if parse_after_refusal:
            page = response.parse()  # reading a paid response is allowed
        with pytest.raises(ApiKeyRefusedError):
            page.get_next_page()
        assert len(wire.paths) == sent
    finally:
        if manager is not None:
            manager.__exit__(None, None, None)
        wrapped.close()


@pytest.mark.parametrize("kind", ["raw", "streaming"])
@pytest.mark.parametrize("parse_after_refusal", [False, True])
async def test_pages_parsed_from_async_envelopes_stay_fenced(
    kind: str, parse_after_refusal: bool
) -> None:
    wire = _Wire()
    wrapped = FakeControlPlane().wrap_async(_async_raw(wire), on_unmetered="allow")
    manager = None

    async def parse(response: Any) -> Any:
        result = response.parse()
        return await result if inspect.isawaitable(result) else result

    try:
        if kind == "raw":
            response = await wrapped.batches.with_raw_response.list()
        else:
            manager = wrapped.batches.with_streaming_response.list()
            response = await manager.__aenter__()
        if not parse_after_refusal:
            page = await parse(response)
        _block(wrapped)
        sent = len(wire.paths)
        if parse_after_refusal:
            page = await parse(response)
        with pytest.raises(ApiKeyRefusedError):
            await page.get_next_page()
        assert len(wire.paths) == sent
    finally:
        if manager is not None:
            await manager.__aexit__(None, None, None)
        await wrapped.close()


def test_an_envelope_reads_without_a_check_and_keeps_its_type() -> None:
    from openai._legacy_response import LegacyAPIResponse

    wire = _Wire()
    wrapped = FakeControlPlane().wrap(_sync_raw(wire), on_unmetered="allow")
    try:
        response = wrapped.batches.with_raw_response.list()
        _block(wrapped)
        assert isinstance(response, LegacyAPIResponse)
        assert response.status_code == 200
        assert response.parse().data[0].id == "batch_1"
    finally:
        wrapped.close()


# ── iter_pages() checks before every page it fetches ──────────────────


@pytest.mark.parametrize("use_yielded_page", [False, True])
def test_sync_iter_pages_checks_every_fetch(use_yielded_page: bool) -> None:
    wire = _Wire()
    wrapped = FakeControlPlane().wrap(_sync_raw(wire), on_unmetered="allow")
    try:
        pages = wrapped.batches.list().iter_pages()
        first = next(pages)
        _block(wrapped)
        sent = len(wire.paths)
        with pytest.raises(ApiKeyRefusedError):
            if use_yielded_page:
                first.get_next_page()
            else:
                next(pages)
        assert len(wire.paths) == sent
    finally:
        wrapped.close()


@pytest.mark.parametrize("use_yielded_page", [False, True])
async def test_async_iter_pages_checks_every_fetch(use_yielded_page: bool) -> None:
    wire = _Wire()
    wrapped = FakeControlPlane().wrap_async(_async_raw(wire), on_unmetered="allow")
    try:
        page = await wrapped.batches.list()
        pages = page.iter_pages()
        first = await pages.__anext__()
        _block(wrapped)
        sent = len(wire.paths)
        with pytest.raises(ApiKeyRefusedError):
            if use_yielded_page:
                await first.get_next_page()
            else:
                await pages.__anext__()
        assert len(wire.paths) == sent
    finally:
        await wrapped.close()
