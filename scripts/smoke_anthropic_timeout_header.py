#!/usr/bin/env python3
"""Smoke-check the per-hop timeout a wrapped anthropic client puts on the wire.

Usage::

    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/smoke_anthropic_timeout_header.py

Makes ONE tiny live Anthropic call through a Solwyn wrapper backed by
``FakeControlPlane`` (so no Solwyn API is needed) and prints only the HTTP
status plus the ``x-stainless-read-timeout`` header and transport timeout that
went out. Anthropic 1.x runs on ``httpx2`` and stringifies a non-``Timeout``
per-request timeout straight into that header, so a healthy run prints the read
bound (e.g. ``600.0``) rather than a tuple repr.

Never prints, stores, or logs prompt or response content. Exits 0 with a skip
message when the key is absent.
"""

from __future__ import annotations

import os
import sys

READ_TIMEOUT_HEADER = "x-stainless-read-timeout"
MODEL = "claude-haiku-4-5"


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("skipped: no ANTHROPIC_API_KEY")
        return 0

    # Imported lazily so the skip path needs neither provider SDK installed.
    import anthropic
    import httpx2

    from solwyn.testing import FakeControlPlane

    observed: dict[str, object] = {}

    def on_request(request: httpx2.Request) -> None:
        observed["header"] = request.headers.get(READ_TIMEOUT_HEADER)
        observed["transport_timeout"] = request.extensions.get("timeout")

    def on_response(response: httpx2.Response) -> None:
        observed["status"] = response.status_code

    hooks = {"request": [on_request], "response": [on_response]}
    provider = anthropic.Anthropic(http_client=httpx2.Client(event_hooks=hooks))
    wrapper = FakeControlPlane().wrap(provider)
    outcome = "ok"
    try:
        wrapper.messages.create(
            model=MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as exc:
        # Class name only: provider error bodies can echo request content.
        outcome = f"error: {type(exc).__name__}"
    finally:
        wrapper.close()

    print(f"outcome: {outcome}")
    print(f"http status: {observed.get('status')}")
    print(f"{READ_TIMEOUT_HEADER}: {observed.get('header')}")
    print(f"transport timeout: {observed.get('transport_timeout')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
