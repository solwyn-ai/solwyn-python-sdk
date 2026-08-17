"""Ownership and interface rules for injected control-plane transports.

An injected transport belongs to the caller.  Each Solwyn-owned HTTP client
therefore receives a forwarding adapter whose close operation is a no-op.  A
default ``None`` transport is passed through unchanged so httpx continues to
create and own its normal transport.

Async control-plane components also perform blocking interpreter-exit drains.
Their injected transport must consequently implement both the sync and async
httpx transport interfaces.
"""

from __future__ import annotations

from typing import Protocol, cast

import httpx


class _SyncTransport(Protocol):
    def handle_request(self, request: httpx.Request) -> httpx.Response: ...


class ControlPlaneTransport(_SyncTransport, Protocol):
    """Transport contract required by async control-plane components."""

    def close(self) -> None: ...

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response: ...

    async def aclose(self) -> None: ...


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
    """Reject a one-sided transport before any async control-plane request.

    The same injected instance is reused when a component repairs itself after
    ``fork()``.  Stateful transports supplied by callers must therefore be
    fork-safe (or confined to processes that do not fork).
    """
    if transport is None:
        return None
    required_methods = ("handle_request", "close", "handle_async_request", "aclose")
    has_both_interfaces = all(
        callable(getattr(transport, method_name, None)) for method_name in required_methods
    )
    sync_handler = getattr(transport, "handle_request", None)
    async_handler = getattr(transport, "handle_async_request", None)
    uses_sync_stub = (
        getattr(sync_handler, "__func__", sync_handler) is httpx.BaseTransport.handle_request
    )
    uses_async_stub = (
        getattr(async_handler, "__func__", async_handler)
        is httpx.AsyncBaseTransport.handle_async_request
    )
    if not has_both_interfaces or uses_sync_stub or uses_async_stub:
        raise TypeError(
            "async control-plane transport must implement both sync and async "
            "httpx transport interfaces"
        )
    return cast("ControlPlaneTransport", transport)


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
