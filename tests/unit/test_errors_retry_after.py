"""Retry-After parsing for the same-provider 429 retry.

``retry_after_seconds`` is duck-typed and status-first like ``classify_exception``:
it returns a non-negative delay (seconds) ONLY for an HTTP 429 carrying a
parseable Retry-After header -- in either RFC 7231 form (delta-seconds or an
HTTP-date) -- and ``None`` otherwise. It never imports a provider SDK; the header
is read off ``exc.response.headers`` (the OpenAI/Anthropic/httpx shape).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from solwyn.providers._errors import retry_after_seconds


def _exc(
    status_code: int, *, retry_after: str | None = None, with_response: bool = True
) -> Exception:
    """Build a duck-typed transport error: ``status_code`` + optional header."""
    exc = Exception("boom")
    exc.status_code = status_code
    if with_response:
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["retry-after"] = retry_after
        exc.response = SimpleNamespace(headers=headers)
    return exc


@pytest.mark.unit
class TestRetryAfterSeconds:
    def test_delta_seconds(self) -> None:
        assert retry_after_seconds(_exc(429, retry_after="2")) == 2.0

    def test_zero_delta(self) -> None:
        assert retry_after_seconds(_exc(429, retry_after="0")) == 0.0

    def test_http_date_future(self) -> None:
        future = datetime.now(UTC) + timedelta(seconds=30)
        delay = retry_after_seconds(_exc(429, retry_after=format_datetime(future)))
        assert delay is not None
        assert 25.0 <= delay <= 31.0

    def test_http_date_in_past_clamps_to_zero(self) -> None:
        past = datetime.now(UTC) - timedelta(seconds=30)
        assert retry_after_seconds(_exc(429, retry_after=format_datetime(past))) == 0.0

    def test_missing_header_returns_none(self) -> None:
        assert retry_after_seconds(_exc(429)) is None

    def test_garbage_header_returns_none(self) -> None:
        assert retry_after_seconds(_exc(429, retry_after="soon")) is None

    def test_non_finite_delta_returns_none(self) -> None:
        assert retry_after_seconds(_exc(429, retry_after="inf")) is None

    def test_non_429_status_returns_none(self) -> None:
        # A 503 (or 529) with a Retry-After is out of scope: only 429 retries.
        assert retry_after_seconds(_exc(503, retry_after="2")) is None
        assert retry_after_seconds(_exc(529, retry_after="2")) is None

    def test_no_response_attr_returns_none(self) -> None:
        assert retry_after_seconds(_exc(429, with_response=False)) is None

    def test_non_canonical_header_casing_on_plain_dict(self) -> None:
        # A plain-dict carrier keyed with a non-canonical casing is still found
        # (real httpx.Headers is case-insensitive; this covers synthetic carriers).
        exc = Exception("boom")
        exc.status_code = 429
        exc.response = SimpleNamespace(headers={"RETRY-AFTER": "5"})
        assert retry_after_seconds(exc) == 5.0

    def test_raising_headers_object_returns_none(self) -> None:
        # The parser runs inside the dispatch except-handler; a pathological
        # headers object whose get() raises must be swallowed (return None), never
        # raised over the original provider exception.
        class _RaisingHeaders:
            def get(self, _name: str) -> str:
                raise RuntimeError("boom")

        exc = Exception("boom")
        exc.status_code = 429
        exc.response = SimpleNamespace(headers=_RaisingHeaders())
        assert retry_after_seconds(exc) is None
