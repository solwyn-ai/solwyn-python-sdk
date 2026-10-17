"""Hermetic StreamingResponse regression with real Starlette context copying."""

from __future__ import annotations

import time

import pytest
from test_otel_population import _raw_client, _response

import solwyn
from solwyn.testing import FakeControlPlane

pytest.importorskip("starlette")
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient


@pytest.mark.unit
@pytest.mark.parametrize("outer", [False, True])
def test_streaming_response_keeps_run_for_all_chunks_and_releases_its_lease(outer):
    plane = FakeControlPlane()
    raw = _raw_client(False)
    raw.chat.completions.create.return_value = _response()
    client = plane.wrap(raw)
    parent_ids = []
    run_ids = []

    class RequestRun:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            with solwyn.run("request") as parent_id:
                parent_ids.append(parent_id)
                await self.app(scope, receive, send)

    def chunks():
        with solwyn.run("chunks") as run_id:
            run_ids.append(run_id)
            for _ in range(3):
                client.chat.completions.create(model="gpt-5.5", messages=[], max_tokens=32)
                yield b"chunk\n"

    async def endpoint(request):
        return StreamingResponse(chunks())

    app = Starlette(
        routes=[Route("/", endpoint)],
        middleware=[Middleware(RequestRun)] if outer else [],
    )
    try:
        with TestClient(app) as browser:
            response = browser.get("/")
        assert response.status_code == 200
        assert response.text == "chunk\n" * 3
        deadline = time.monotonic() + 3
        while not plane.lease_surrenders and time.monotonic() < deadline:
            time.sleep(0.005)
        # The stream scope releases while the wrapper is still open.
        assert [s.lease_id for s in plane.lease_surrenders] == ["lse_fake1"]
    finally:
        client.close()
    parent = parent_ids[-1] if outer else None
    assert [(e.agent_run_id, e.parent_agent_run_id) for e in plane.ingested] == [
        (run_ids[0], parent)
    ] * 3
    assert [g.agent_run_id for g in plane.lease_grants] == run_ids
    assert len(plane.lease_surrenders) == 1
