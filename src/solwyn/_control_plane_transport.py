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


class _DualTransport(_SyncTransport, Protocol):
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

    def __init__(self, transport: _DualTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        """Leave the caller-owned backing transport open."""


def require_dual_transport(
    transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None,
) -> _DualTransport | None:
    """Reject a one-sided transport before any async control-plane request.

    The same injected instance is reused when a component repairs itself after
    ``fork()``.  Stateful transports supplied by callers must therefore be
    fork-safe (or confined to processes that do not fork).
    """
    if transport is None:
        return None
    if not isinstance(transport, httpx.BaseTransport) or not isinstance(
        transport, httpx.AsyncBaseTransport
    ):
        raise TypeError(
            "async control-plane transport must implement both sync and async "
            "httpx transport interfaces"
        )
    return cast("_DualTransport", transport)


def non_closing_sync_transport(
    transport: _SyncTransport | None,
) -> httpx.BaseTransport | None:
    """Return an httpx sync transport without transferring caller ownership."""
    if transport is None:
        return None
    return _NonClosingSyncTransport(transport)


def non_closing_async_transport(
    transport: _DualTransport | None,
) -> httpx.AsyncBaseTransport | None:
    """Return an httpx async transport without transferring caller ownership."""
    if transport is None:
        return None
    return _NonClosingAsyncTransport(transport)
