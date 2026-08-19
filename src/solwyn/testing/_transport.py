"""Dual sync/async httpx transport for the fake control plane."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Protocol

import httpx

from solwyn.testing._wire import PlaneResponse, PreparedPlaneRequest, serialize_response


class _Plane(Protocol):
    def _prepare_request(
        self,
        method: str,
        path: str,
        body: object,
        parse_error: PlaneResponse | None,
    ) -> PreparedPlaneRequest: ...


class FakeControlPlaneTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Route httpx requests directly into a ``FakeControlPlane`` instance."""

    def __init__(self, plane: _Plane) -> None:
        self._plane = plane

    @staticmethod
    def _decode_body(request: httpx.Request) -> tuple[object, PlaneResponse | None]:
        body: object = None
        if request.content:
            try:
                body = json.loads(request.content)
            except (TypeError, ValueError):
                return None, PlaneResponse(
                    422,
                    {
                        "detail": [
                            {
                                "type": "json_invalid",
                                "loc": ["body"],
                                "msg": "JSON decode error",
                            }
                        ]
                    },
                )
        return body, None

    def _prepare(self, request: httpx.Request) -> PreparedPlaneRequest:
        body, parse_error = self._decode_body(request)
        return self._plane._prepare_request(
            request.method,
            request.url.path,
            body,
            parse_error,
        )

    @staticmethod
    def _response(request: httpx.Request, result: PlaneResponse) -> httpx.Response:
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
        prepared = self._prepare(request)
        if prepared.delay_seconds:
            time.sleep(prepared.delay_seconds)
        if prepared.outage:
            raise httpx.ConnectError(
                "solwyn.testing: scripted outage",
                request=request,
            )
        if prepared.response is None:
            raise RuntimeError("solwyn.testing plane prepared no response")
        return self._response(request, prepared.response)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        prepared = self._prepare(request)
        if prepared.delay_seconds:
            await asyncio.sleep(prepared.delay_seconds)
        if prepared.outage:
            raise httpx.ConnectError(
                "solwyn.testing: scripted outage",
                request=request,
            )
        if prepared.response is None:
            raise RuntimeError("solwyn.testing plane prepared no response")
        return self._response(request, prepared.response)

    def close(self) -> None:
        """The in-memory transport owns no resources."""

    async def aclose(self) -> None:
        """The in-memory transport owns no resources."""
