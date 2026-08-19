"""Ownership and interface rules for injected control-plane transports.

An injected transport belongs to the caller.  Each Solwyn-owned HTTP client
therefore receives a forwarding adapter whose close operation is a no-op, and
that adapter's close/aclose is never forwarded to the backing transport — so
neither validator below requires the transport to implement close/aclose at
all.  A default ``None`` transport is passed through unchanged so httpx
continues to create and own its normal transport.  ``require_sync_transport``
validates a transport for sync-only components (``BudgetEnforcer``,
``MetadataReporter``, sync ``Solwyn``); ``require_dual_transport`` validates
one for async components.

Async control-plane components also perform blocking interpreter-exit drains.
Their injected transport must consequently implement both methods the SDK
actually calls: a callable, non-coroutine ``handle_request`` and a coroutine
``handle_async_request``.
"""

from __future__ import annotations

import inspect
from typing import Protocol, cast

import httpx


class _SyncTransport(Protocol):
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...


class ControlPlaneTransport(_SyncTransport, Protocol):
    """Transport contract required by async control-plane components.

    Only the two methods the SDK actually calls: a callable, non-coroutine
    ``handle_request`` (inherited from :class:`_SyncTransport`) and a
    coroutine ``handle_async_request``. ``close``/``aclose`` are deliberately
    absent — the non-closing wrappers never forward them to a caller-owned
    transport, so requiring them would reject a functionally sufficient
    transport with a misleading error.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response: ...


class _NonClosingSyncTransport(httpx.BaseTransport):
    """Forward sync requests without taking ownership of the backing transport."""

    def __init__(self, transport: _SyncTransport) -> None:
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._transport.handle_request(request)

    def close(self) -> None:
        """Leave the caller-owned backing transport open."""


class _NonClosingAsyncTransport(httpx.AsyncBaseTransport):
    """Forward async requests without taking ownership of the backing transport."""

    def __init__(self, transport: ControlPlaneTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        """Leave the caller-owned backing transport open."""


def require_dual_transport(
    transport: object | None,
) -> ControlPlaneTransport | None:
    """Reject a transport missing either method an async control-plane request needs.

    Only ``handle_request`` and ``handle_async_request`` are checked —
    ``close``/``aclose`` are never called on a caller-owned transport (the
    non-closing wrappers swallow them), so a transport lacking them is still
    functionally sufficient and must not be rejected.

    The same injected instance is reused when a component repairs itself after
    ``fork()``.  Stateful transports supplied by callers must therefore be
    fork-safe (or confined to processes that do not fork).
    """
    if transport is None:
        return None
    sync_handler = getattr(transport, "handle_request", None)
    async_handler = getattr(transport, "handle_async_request", None)
    has_both_interfaces = callable(sync_handler) and callable(async_handler)
    uses_sync_stub = (
        getattr(sync_handler, "__func__", sync_handler) is httpx.BaseTransport.handle_request
    )
    uses_async_stub = (
        getattr(async_handler, "__func__", async_handler)
        is httpx.AsyncBaseTransport.handle_async_request
    )
    has_correct_method_shapes = not inspect.iscoroutinefunction(
        sync_handler
    ) and inspect.iscoroutinefunction(async_handler)
    if (
        not has_both_interfaces
        or uses_sync_stub
        or uses_async_stub
        or not has_correct_method_shapes
    ):
        raise TypeError(
            "async control-plane transport must implement both sync and async "
            "httpx transport interfaces (a callable, non-coroutine handle_request "
            "and a coroutine handle_async_request) with the correct sync and "
            "async method shapes"
        )
    return cast("ControlPlaneTransport", transport)


def require_sync_transport(
    transport: object | None,
) -> _SyncTransport | None:
    """Reject a transport that cannot serve a sync control-plane request.

    Sync control-plane components (``BudgetEnforcer``, ``MetadataReporter``,
    sync ``Solwyn``) only ever call ``handle_request`` — unlike
    :func:`require_dual_transport`, which additionally requires
    ``handle_async_request``, this checks the sync method alone. Neither
    validator requires ``close``/``aclose``.
    """
    if transport is None:
        return None
    handler = getattr(transport, "handle_request", None)
    uses_stub = getattr(handler, "__func__", handler) is httpx.BaseTransport.handle_request
    if not callable(handler) or inspect.iscoroutinefunction(handler) or uses_stub:
        raise TypeError(
            "sync control-plane transport must implement the httpx sync "
            "transport interface (a callable, non-async handle_request)"
        )
    return cast("_SyncTransport", transport)


def non_closing_sync_transport(
    transport: _SyncTransport | None,
) -> httpx.BaseTransport | None:
    """Return an httpx sync transport without transferring caller ownership."""
    if transport is None:
        return None
    return _NonClosingSyncTransport(transport)


def non_closing_async_transport(
    transport: ControlPlaneTransport | None,
) -> httpx.AsyncBaseTransport | None:
    """Return an httpx async transport without transferring caller ownership."""
    if transport is None:
        return None
    return _NonClosingAsyncTransport(transport)
