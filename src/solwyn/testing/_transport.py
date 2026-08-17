"""Dual sync/async httpx transport for the fake control plane."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Protocol

import httpx

from solwyn.testing._wire import PlaneResponse, serialize_response


class _Plane(Protocol):
    def handle(self, method: str, path: str, body: object) -> PlaneResponse: ...

    def _transport_effect(self, method: str, path: str) -> tuple[float, bool]: ...


class FakeControlPlaneTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Route httpx requests directly into a ``FakeControlPlane`` instance."""

    def __init__(self, plane: _Plane) -> None:
        self._plane = plane

    def _response_after_effect(self, request: httpx.Request) -> httpx.Response:
        body: object = None
        if request.content:
            try:
                body = json.loads(request.content)
            except (TypeError, ValueError):
                return httpx.Response(
                    422,
                    request=request,
                    json={
                        "detail": [
                            {
                                "type": "json_invalid",
                                "loc": ["body"],
                                "msg": "JSON decode error",
                            }
                        ]
                    },
                )
        result = self._plane.handle(request.method, request.url.path, body)
        serialized = serialize_response(result)
        if serialized is None:
            return httpx.Response(
                result.status_code,
                request=request,
                headers=result.headers,
            )
        return httpx.Response(
            result.status_code,
            request=request,
            headers=result.headers,
            json=serialized,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        delay, outage = self._plane._transport_effect(request.method, request.url.path)
        if delay:
            time.sleep(delay)
        if outage:
            raise httpx.ConnectError(
                "solwyn.testing: scripted outage",
                request=request,
            )
        return self._response_after_effect(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        delay, outage = self._plane._transport_effect(request.method, request.url.path)
        if delay:
            await asyncio.sleep(delay)
        if outage:
            raise httpx.ConnectError(
                "solwyn.testing: scripted outage",
                request=request,
            )
        return self._response_after_effect(request)

    def close(self) -> None:
        """The in-memory transport owns no resources."""

    async def aclose(self) -> None:
        """The in-memory transport owns no resources."""
