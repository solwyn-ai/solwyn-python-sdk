"""Tests for status-first, duck-typed transport-error classification.

These tests mirror the real provider-SDK exception subclassing without importing
any provider SDK (openai/anthropic/google/boto3 are NOT installed). The critical
invariant under test is the ORDERING trap: ``APITimeoutError`` subclasses
``APIConnectionError`` in both openai and anthropic, so a timeout (post-send,
unsafe) must be matched BEFORE the bare connection-failure (pre-send, safe)
branch — otherwise a silent double-spend results.

The botocore fakes mirror botocore/exceptions.py exactly: ``ReadTimeoutError``
subclasses ``HTTPClientError`` (NOT botocore's ``ConnectionError``);
``EndpointConnectionError`` / ``ConnectTimeoutError`` subclass botocore's
``ConnectionError`` — whose class NAME collides with the Python builtin, which
is why the classifier must never match the bare name "ConnectionError".
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from solwyn.providers._errors import Disposition, classify_exception, retry_after_seconds

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
# 8. botocore / Bedrock classification (duck-typed; boto3 never imported)
# ---------------------------------------------------------------------------


# Mirror the real botocore hierarchy (verified against botocore/exceptions.py):
#   ConnectionError(BotoCoreError)            <- name collides with the builtin!
#   EndpointConnectionError(ConnectionError)
#   ConnectTimeoutError(ConnectionError)
#   HTTPClientError(BotoCoreError)
#   ReadTimeoutError(HTTPClientError)         <- NOT under ConnectionError
#   ConnectionClosedError(HTTPClientError)
class _BotoCoreError(Exception):
    pass


_BotocoreConnectionError = type("ConnectionError", (_BotoCoreError,), {})
_EndpointConnectionError = type("EndpointConnectionError", (_BotocoreConnectionError,), {})
_ConnectTimeoutError = type("ConnectTimeoutError", (_BotocoreConnectionError,), {})
_HTTPClientError = type("HTTPClientError", (_BotoCoreError,), {})
_ReadTimeoutError = type("ReadTimeoutError", (_HTTPClientError,), {})
_ConnectionClosedError = type("ConnectionClosedError", (_HTTPClientError,), {})


class _ClientError(Exception):
    """Mirrors botocore.exceptions.ClientError: status lives in a response DICT."""

    def __init__(self, error_response: dict[str, Any], operation_name: str = "Converse") -> None:
        super().__init__("An error occurred")
        self.response = error_response
        self.operation_name = operation_name


def _bedrock_client_error(
    code: str, status: int, headers: dict[str, str] | None = None
) -> Exception:
    """Build a fake Bedrock service exception.

    botocore generates one ClientError subclass per modeled error shape, with
    the class NAME equal to the error code — mirror that exactly.
    """
    response: dict[str, Any] = {
        "Error": {"Code": code, "Message": "boom"},
        "ResponseMetadata": {"HTTPStatusCode": status, "HTTPHeaders": headers or {}},
    }
    cls = type(code, (_ClientError,), {})
    return cls(response)


@pytest.mark.unit
class TestBotocoreStatusClassification:
    """Bedrock service errors carry status in exc.response, not exc.status_code."""

    def test_throttling_exception_429_is_failover(self) -> None:
        exc = _bedrock_client_error("ThrottlingException", 429)
        assert classify_exception(exc) is Disposition.FAILOVER

    def test_model_not_ready_429_is_failover(self) -> None:
        exc = _bedrock_client_error("ModelNotReadyException", 429)
        assert classify_exception(exc) is Disposition.FAILOVER

    @pytest.mark.parametrize(
        "code,status",
        [
            ("ValidationException", 400),
            ("AccessDeniedException", 403),
            ("ResourceNotFoundException", 404),
        ],
    )
    def test_request_shaped_4xx_is_fail_fast(self, code: str, status: int) -> None:
        assert classify_exception(_bedrock_client_error(code, status)) is Disposition.FAIL_FAST

    @pytest.mark.parametrize(
        "code,status",
        [("InternalServerException", 500), ("ServiceUnavailableException", 503)],
    )
    def test_5xx_is_ambiguous(self, code: str, status: int) -> None:
        assert (
            classify_exception(_bedrock_client_error(code, status))
            is Disposition.POST_SEND_AMBIGUOUS
        )


@pytest.mark.unit
class TestBedrockNameOverridesStatus:
    """Bedrock's post-send errors carry 4xx statuses that would misclassify.

    ModelTimeoutException is HTTP 408 and ModelErrorException is HTTP 424 —
    by status alone both would be FAIL_FAST (request-shaped), but the request
    REACHED the model and ran. They must classify POST_SEND_AMBIGUOUS so the
    reservation reconciles instead of silently double-counting.
    """

    def test_model_timeout_408_is_ambiguous_not_fail_fast(self) -> None:
        exc = _bedrock_client_error("ModelTimeoutException", 408)
        assert classify_exception(exc) is Disposition.POST_SEND_AMBIGUOUS

    def test_model_error_424_is_ambiguous_not_fail_fast(self) -> None:
        exc = _bedrock_client_error("ModelErrorException", 424)
        assert classify_exception(exc) is Disposition.POST_SEND_AMBIGUOUS


@pytest.mark.unit
class TestBotocoreTransportClassification:
    """botocore transport errors classify by MRO name (no status anywhere)."""

    def test_endpoint_connection_error_is_failover(self) -> None:
        # Provably pre-send: could not connect to the endpoint at all.
        assert classify_exception(_EndpointConnectionError()) is Disposition.FAILOVER

    def test_connect_timeout_error_is_failover(self) -> None:
        assert classify_exception(_ConnectTimeoutError()) is Disposition.FAILOVER

    def test_read_timeout_error_is_ambiguous(self) -> None:
        # Post-send: bytes were sent, the model may have run.
        assert classify_exception(_ReadTimeoutError()) is Disposition.POST_SEND_AMBIGUOUS

    def test_connection_closed_error_is_ambiguous(self) -> None:
        # Connection dropped before a valid response — may be mid-response.
        assert classify_exception(_ConnectionClosedError()) is Disposition.POST_SEND_AMBIGUOUS

    def test_builtin_connection_reset_stays_fail_fast(self) -> None:
        # The name-collision guard: botocore's ConnectionError shares its NAME
        # with the builtin, and builtin ConnectionResetError (post-send
        # possible) has "ConnectionError" in its MRO names. The classifier must
        # never match the bare name — a reset stays at the safe default.
        assert classify_exception(ConnectionResetError()) is Disposition.FAIL_FAST


@pytest.mark.unit
class TestBotocoreRetryAfter:
    def test_retry_after_read_from_response_metadata_http_headers(self) -> None:
        exc = _bedrock_client_error("ThrottlingException", 429, headers={"retry-after": "2"})
        assert retry_after_seconds(exc) == 2.0

    def test_no_header_returns_none(self) -> None:
        exc = _bedrock_client_error("ThrottlingException", 429)
        assert retry_after_seconds(exc) is None

    def test_non_429_returns_none_even_with_header(self) -> None:
        exc = _bedrock_client_error(
            "ServiceUnavailableException", 503, headers={"retry-after": "2"}
        )
        assert retry_after_seconds(exc) is None


# ---------------------------------------------------------------------------
# 9. APIConnectionError cause-inspection (the subtle post-send double-spend
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
