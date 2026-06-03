"""Tests for status-first, duck-typed transport-error classification (§6.1).

These tests mirror the real provider-SDK exception subclassing without importing
any provider SDK (openai/anthropic/google are NOT installed). The critical
invariant under test is the ORDERING trap: ``APITimeoutError`` subclasses
``APIConnectionError`` in both openai and anthropic, so a timeout (post-send,
unsafe) must be matched BEFORE the bare connection-failure (pre-send, safe)
branch — otherwise a silent double-spend results.
"""

from __future__ import annotations

import httpx
import pytest

from solwyn.providers._errors import Disposition, classify_exception

# ---------------------------------------------------------------------------
# Fake provider exceptions — mirror the real subclassing WITHOUT importing SDKs.
# ---------------------------------------------------------------------------


class APIConnectionError(Exception):
    """Mirrors openai/anthropic APIConnectionError (bare connection failure)."""


class APITimeoutError(APIConnectionError):
    """Mirrors openai/anthropic APITimeoutError — SUBCLASSES APIConnectionError.

    This is the double-spend trap: a naive ``except APIConnectionError`` would
    catch this post-send timeout and (wrongly) failover.
    """


class _Status(Exception):
    """Fake openai/anthropic API error carrying a numeric ``.status_code``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status={status_code}")
        self.status_code = status_code


class _GoogleErr(Exception):
    """Fake google APIError carrying a numeric ``.code`` (Google's status field)."""

    def __init__(self, code: int) -> None:
        super().__init__(f"code={code}")
        self.code = code


# ---------------------------------------------------------------------------
# 1. The double-spend ordering trap (APITimeoutError ⊂ APIConnectionError)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrderingTrap:
    def test_api_timeout_error_is_ambiguous_despite_subclassing_connection_error(
        self,
    ) -> None:
        # Arrange: APITimeoutError IS-A APIConnectionError (verify the trap exists)
        assert issubclass(APITimeoutError, APIConnectionError)
        exc = APITimeoutError("read timed out")

        # Act
        disp = classify_exception(exc)

        # Assert: timeout wins -> post-send ambiguous, NOT failover
        assert disp is Disposition.POST_SEND_AMBIGUOUS

    def test_bare_api_connection_error_is_failover(self) -> None:
        # Arrange: bare connection failure, no status, not a timeout
        exc = APIConnectionError("connection refused")

        # Act
        disp = classify_exception(exc)

        # Assert: provably pre-send -> safe to failover
        assert disp is Disposition.FAILOVER


# ---------------------------------------------------------------------------
# 2. httpx transport errors (Google pre-send failures + ambiguous reads)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHttpxTransportErrors:
    def test_read_timeout_is_ambiguous(self) -> None:
        # ReadTimeout may have reached the server -> post-send ambiguous
        assert (
            classify_exception(httpx.ReadTimeout("read timed out"))
            is Disposition.POST_SEND_AMBIGUOUS
        )

    def test_write_timeout_is_ambiguous(self) -> None:
        assert (
            classify_exception(httpx.WriteTimeout("write timed out"))
            is Disposition.POST_SEND_AMBIGUOUS
        )

    def test_connect_timeout_is_failover(self) -> None:
        # ConnectTimeout subclasses TimeoutException but is provably pre-send.
        # Verify the subclassing so the ordering is meaningfully tested.
        assert issubclass(httpx.ConnectTimeout, httpx.TimeoutException)
        assert classify_exception(httpx.ConnectTimeout("connect timed out")) is Disposition.FAILOVER

    def test_pool_timeout_is_failover(self) -> None:
        # PoolTimeout means we never acquired a connection -> pre-send.
        assert classify_exception(httpx.PoolTimeout("pool timed out")) is Disposition.FAILOVER

    def test_connect_error_is_failover(self) -> None:
        # Connection refused / DNS / TLS -> never sent -> safe to failover.
        assert classify_exception(httpx.ConnectError("connection refused")) is Disposition.FAILOVER

    def test_generic_timeout_exception_is_ambiguous(self) -> None:
        # Any other TimeoutException we cannot prove pre-send -> ambiguous.
        assert (
            classify_exception(httpx.TimeoutException("unknown timeout"))
            is Disposition.POST_SEND_AMBIGUOUS
        )

    def test_remote_protocol_error_is_ambiguous(self) -> None:
        # Mid-flight transport error (e.g. server dropped the connection).
        assert (
            classify_exception(httpx.RemoteProtocolError("peer closed connection"))
            is Disposition.POST_SEND_AMBIGUOUS
        )

    def test_generic_transport_error_is_ambiguous(self) -> None:
        assert (
            classify_exception(httpx.TransportError("transport boom"))
            is Disposition.POST_SEND_AMBIGUOUS
        )


# ---------------------------------------------------------------------------
# 3. Numeric status classification (OpenAI/Anthropic .status_code)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStatusCodeClassification:
    @pytest.mark.parametrize("status", [429, 529])
    def test_rate_limit_and_overloaded_are_failover(self, status: int) -> None:
        # 429 rate limit, 529 Anthropic overloaded -> pre-send rejection.
        assert classify_exception(_Status(status)) is Disposition.FAILOVER

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_4xx_is_fail_fast(self, status: int) -> None:
        # Same error everywhere -> stop the chain, never failover.
        assert classify_exception(_Status(status)) is Disposition.FAIL_FAST

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_5xx_is_ambiguous(self, status: int) -> None:
        # Response received; can't tell if model ran -> never failover.
        assert classify_exception(_Status(status)) is Disposition.POST_SEND_AMBIGUOUS


# ---------------------------------------------------------------------------
# 4. Google numeric status via .code
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoogleCodeClassification:
    def test_google_429_is_failover(self) -> None:
        # Google APIError keys on .code, not .status_code.
        assert classify_exception(_GoogleErr(429)) is Disposition.FAILOVER

    def test_google_503_is_ambiguous(self) -> None:
        # Google ServerError -> no idempotency key -> re-send double-charges.
        assert classify_exception(_GoogleErr(503)) is Disposition.POST_SEND_AMBIGUOUS

    @pytest.mark.parametrize("code", [400, 404])
    def test_google_4xx_is_fail_fast(self, code: int) -> None:
        assert classify_exception(_GoogleErr(code)) is Disposition.FAIL_FAST


# ---------------------------------------------------------------------------
# 5. bool guard — bool subclasses int, must NOT be read as a status code
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBoolGuard:
    def test_bool_status_code_is_not_treated_as_numeric_status(self) -> None:
        # Arrange: an exc whose .status_code is a bool (bool ⊂ int).
        class _BoolStatus(Exception):
            status_code = True  # noqa: F841  (truthy bool, == 1 if read as int)

        # Act
        disp = classify_exception(_BoolStatus())

        # Assert: bool is ignored -> falls through to default FAIL_FAST,
        # NOT misread as HTTP 1.
        assert disp is Disposition.FAIL_FAST


# ---------------------------------------------------------------------------
# 6. Default — unknown exceptions never failover into the unknown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefault:
    def test_value_error_is_fail_fast(self) -> None:
        assert classify_exception(ValueError("nope")) is Disposition.FAIL_FAST

    def test_base_exception_is_fail_fast(self) -> None:
        # Signature accepts BaseException; a non-Exception still fails fast.
        assert classify_exception(KeyboardInterrupt()) is Disposition.FAIL_FAST


# ---------------------------------------------------------------------------
# 7. Disposition enum surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDispositionEnum:
    def test_string_values(self) -> None:
        # StrEnum: each member IS its string value.
        assert Disposition.FAILOVER == "failover"
        assert Disposition.POST_SEND_AMBIGUOUS == "post_send_ambiguous"
        assert Disposition.FAIL_FAST == "fail_fast"


# ---------------------------------------------------------------------------
# 8. APIConnectionError cause-inspection (the subtle post-send double-spend
#    vector): the provider wrapper covers BOTH pre-send and post-send transport
#    failures, so the name alone is NOT proof of pre-send — classify by cause.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApiConnectionErrorCauseInspection:
    def test_cause_connect_error_is_failover(self) -> None:
        exc = APIConnectionError("connection refused")
        exc.__cause__ = httpx.ConnectError("refused")
        assert classify_exception(exc) is Disposition.FAILOVER

    def test_cause_connect_timeout_is_failover(self) -> None:
        exc = APIConnectionError("connect timed out")
        exc.__cause__ = httpx.ConnectTimeout("timeout")
        assert classify_exception(exc) is Disposition.FAILOVER

    def test_cause_remote_protocol_error_is_ambiguous(self) -> None:
        # Server disconnected mid-response: the request may have run -> ambiguous.
        exc = APIConnectionError("server disconnected")
        exc.__cause__ = httpx.RemoteProtocolError("peer closed connection")
        assert classify_exception(exc) is Disposition.POST_SEND_AMBIGUOUS

    def test_cause_read_timeout_is_ambiguous(self) -> None:
        exc = APIConnectionError("read error")
        exc.__cause__ = httpx.ReadTimeout("slow")
        assert classify_exception(exc) is Disposition.POST_SEND_AMBIGUOUS

    def test_context_chained_cause_is_inspected(self) -> None:
        # Implicit chaining (__context__) is honored when __cause__ is unset.
        exc = APIConnectionError("server disconnected")
        exc.__context__ = httpx.RemoteProtocolError("peer closed connection")
        assert classify_exception(exc) is Disposition.POST_SEND_AMBIGUOUS
