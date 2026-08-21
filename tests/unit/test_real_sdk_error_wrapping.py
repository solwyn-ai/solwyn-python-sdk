"""Real-SDK error-wrapping tests — genuine provider clients, real exceptions.

Every other classification test in this suite hand-rolls a synthetic exception
class named ``APITimeoutError`` (see test_failover_idempotency.py) — convenient,
and correct for pinning the name-matching contract, but structurally BLIND to
what the real SDKs actually put in ``__cause__``. That blindness hid a P1: both
openai and anthropic catch the whole ``httpx.TimeoutException`` family and
re-raise it as ``APITimeoutError``, so a provably PRE-send ``ConnectTimeout`` /
``PoolTimeout`` arrives wearing the class name of a POST-send read timeout.
Classifying by name alone stranded every connect-slice timeout on a dead
primary — the chain could never advance under default-safe idempotency.

These tests drive REAL ``openai`` / ``anthropic`` clients over the matching
``httpx.MockTransport`` / ``httpx2.MockTransport`` that raises a chosen
transport error, then assert Solwyn's disposition for the exception the SDK
actually produced. Hermetic: MockTransport never opens a socket. Each SDK is
gated by its own ``pytest.importorskip`` fixture so a missing package skips only
its own tests.

This matters more since PJ-8/R7 split connect from read: the connect slice is
short and deadline-derived, so on a stalled provider it is the bound that fires
FIRST — the pre-send case is the common one, not the corner.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from solwyn.providers._errors import Disposition, classify_exception

VALID_OPENAI_KEY = "sk-test-not-a-real-key"
VALID_ANTHROPIC_KEY = "sk-ant-test-not-a-real-key"


@pytest.fixture(scope="module")
def openai_mod() -> Any:
    return pytest.importorskip("openai")


@pytest.fixture(scope="module")
def anthropic_mod() -> Any:
    return pytest.importorskip("anthropic")


@pytest.fixture(scope="module")
def anthropic_httpx_mod() -> Any:
    """Anthropic 1.x ships on httpx2 rather than the app's httpx runtime."""
    return pytest.importorskip("httpx2")


def _raising_transport(transport_mod: Any, exc: Exception) -> Any:
    """A transport that raises ``exc`` instead of performing any I/O."""

    def handler(request: Any) -> Any:
        raise exc

    return transport_mod.MockTransport(handler)


def _openai_failure(openai_mod: Any, exc: Exception) -> BaseException:
    """Return the exception a REAL openai client raises for this httpx error."""
    client = openai_mod.OpenAI(
        api_key=VALID_OPENAI_KEY,
        http_client=httpx.Client(transport=_raising_transport(httpx, exc)),
        max_retries=0,
    )
    with pytest.raises(Exception) as exc_info:  # noqa: PT011 - the SDK class is the subject
        client.chat.completions.create(
            model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
        )
    return exc_info.value


def _anthropic_failure(
    anthropic_mod: Any,
    anthropic_httpx_mod: Any,
    exc: Exception,
) -> BaseException:
    """Return the exception a REAL anthropic client raises for this httpx error."""
    client = anthropic_mod.Anthropic(
        api_key=VALID_ANTHROPIC_KEY,
        http_client=anthropic_httpx_mod.Client(
            transport=_raising_transport(anthropic_httpx_mod, exc)
        ),
        max_retries=0,
    )
    with pytest.raises(Exception) as exc_info:  # noqa: PT011 - the SDK class is the subject
        client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
    return exc_info.value


# The load-bearing table: what the SDK wrapper hides, and what it must classify
# as anyway. Pre-send causes are failover-safe; post-send causes must NOT
# advance the chain (a re-attempt could double-spend a request that already ran).
_TIMEOUT_CAUSES = [
    pytest.param(httpx.ConnectTimeout("connect stalled"), Disposition.FAILOVER, id="connect"),
    pytest.param(httpx.PoolTimeout("pool exhausted"), Disposition.FAILOVER, id="pool"),
    pytest.param(httpx.ReadTimeout("read stalled"), Disposition.POST_SEND_AMBIGUOUS, id="read"),
    pytest.param(httpx.WriteTimeout("write stalled"), Disposition.POST_SEND_AMBIGUOUS, id="write"),
]

_ANTHROPIC_TIMEOUT_CAUSES = [
    pytest.param("ConnectTimeout", Disposition.FAILOVER, id="connect"),
    pytest.param("PoolTimeout", Disposition.FAILOVER, id="pool"),
    pytest.param("ReadTimeout", Disposition.POST_SEND_AMBIGUOUS, id="read"),
    pytest.param("WriteTimeout", Disposition.POST_SEND_AMBIGUOUS, id="write"),
]


@pytest.mark.unit
class TestRealSdkTimeoutWrapping:
    """The APITimeoutError wrapper must be classified by its CAUSE, not its name."""

    @pytest.mark.parametrize(("cause", "expected"), _TIMEOUT_CAUSES)
    def test_openai_timeout_classified_by_cause(
        self, openai_mod: Any, cause: Exception, expected: Disposition
    ) -> None:
        # Arrange + Act: a real openai client meets a real httpx timeout.
        raised = _openai_failure(openai_mod, cause)

        # Assert: the SDK flattens all four into one class — the contract this
        # test defends is that Solwyn does NOT.
        assert type(raised).__name__ == "APITimeoutError"
        assert type(raised.__cause__) is type(cause)
        assert classify_exception(raised) is expected

    @pytest.mark.parametrize(("cause_name", "expected"), _ANTHROPIC_TIMEOUT_CAUSES)
    def test_anthropic_timeout_classified_by_cause(
        self,
        anthropic_mod: Any,
        anthropic_httpx_mod: Any,
        cause_name: str,
        expected: Disposition,
    ) -> None:
        cause_type = getattr(anthropic_httpx_mod, cause_name)
        cause = cause_type(f"{cause_name} from Anthropic's transport")
        raised = _anthropic_failure(anthropic_mod, anthropic_httpx_mod, cause)

        assert type(raised).__name__ == "APITimeoutError"
        assert type(raised.__cause__) is cause_type
        assert classify_exception(raised) is expected

    def test_openai_connect_error_still_failover(self, openai_mod: Any) -> None:
        # The non-timeout pre-send case rides the APIConnectionError branch;
        # pinned here so the shared pre-send tuple can't regress for one wrapper
        # while passing for the other.
        raised = _openai_failure(openai_mod, httpx.ConnectError("refused"))

        assert type(raised).__name__ == "APIConnectionError"
        assert classify_exception(raised) is Disposition.FAILOVER

    def test_openai_remote_protocol_error_stays_ambiguous(self, openai_mod: Any) -> None:
        # A server disconnect MID-RESPONSE also arrives as APIConnectionError.
        # It is post-send: the model may have already run and charged.
        raised = _openai_failure(openai_mod, httpx.RemoteProtocolError("server disconnected"))

        assert type(raised).__name__ == "APIConnectionError"
        assert classify_exception(raised) is Disposition.POST_SEND_AMBIGUOUS

    def test_anthropic_connect_error_still_failover(
        self,
        anthropic_mod: Any,
        anthropic_httpx_mod: Any,
    ) -> None:
        cause = anthropic_httpx_mod.ConnectError("refused")
        raised = _anthropic_failure(anthropic_mod, anthropic_httpx_mod, cause)

        assert type(raised).__name__ == "APIConnectionError"
        assert type(raised.__cause__) is anthropic_httpx_mod.ConnectError
        assert classify_exception(raised) is Disposition.FAILOVER

    def test_anthropic_remote_protocol_error_stays_ambiguous(
        self,
        anthropic_mod: Any,
        anthropic_httpx_mod: Any,
    ) -> None:
        cause = anthropic_httpx_mod.RemoteProtocolError("server disconnected")
        raised = _anthropic_failure(anthropic_mod, anthropic_httpx_mod, cause)

        assert type(raised).__name__ == "APIConnectionError"
        assert type(raised.__cause__) is anthropic_httpx_mod.RemoteProtocolError
        assert classify_exception(raised) is Disposition.POST_SEND_AMBIGUOUS

    @pytest.mark.parametrize(
        ("cause_name", "expected"),
        [
            pytest.param("ConnectError", Disposition.FAILOVER, id="connect-error"),
            pytest.param(
                "RemoteProtocolError",
                Disposition.POST_SEND_AMBIGUOUS,
                id="remote-protocol",
            ),
        ],
    )
    def test_direct_anthropic_transport_error_uses_send_certainty(
        self,
        anthropic_httpx_mod: Any,
        cause_name: str,
        expected: Disposition,
    ) -> None:
        cause = getattr(anthropic_httpx_mod, cause_name)("direct transport failure")

        assert classify_exception(cause) is expected

    def test_external_connect_error_name_cannot_override_real_httpx2_ancestry(
        self,
        anthropic_httpx_mod: Any,
    ) -> None:
        class ConnectError(anthropic_httpx_mod.RemoteProtocolError):
            pass

        cause = ConnectError("post-send protocol failure with a misleading subclass name")

        assert classify_exception(cause) is Disposition.POST_SEND_AMBIGUOUS


@pytest.mark.unit
class TestSyntheticWrapperDefaults:
    """The no-cause fallbacks — deliberately OPPOSITE between the two wrappers."""

    def test_bare_timeout_wrapper_defaults_to_ambiguous(self) -> None:
        # A bare timeout is canonically a READ timeout: never failover.
        class APITimeoutError(Exception):
            pass

        assert classify_exception(APITimeoutError("timed out")) is Disposition.POST_SEND_AMBIGUOUS

    def test_bare_connection_wrapper_defaults_to_failover(self) -> None:
        # A bare connection error is canonically connect-refused: the outage
        # case failover exists to handle.
        class APIConnectionError(Exception):
            pass

        assert classify_exception(APIConnectionError("connection error")) is Disposition.FAILOVER

    def test_timeout_wrapper_with_non_httpx_cause_stays_ambiguous(self) -> None:
        # Only a recognized pre-send httpx class earns failover — an arbitrary
        # chained cause proves nothing about whether the request landed.
        class APITimeoutError(Exception):
            pass

        exc = APITimeoutError("timed out")
        exc.__cause__ = ValueError("something else entirely")
        assert classify_exception(exc) is Disposition.POST_SEND_AMBIGUOUS

    def test_synthetic_httpx2_module_strings_do_not_admit_a_transport_family(self) -> None:
        class TransportError(Exception):
            pass

        class ConnectError(TransportError):
            pass

        TransportError.__module__ = "httpx2"
        ConnectError.__module__ = "httpx2"

        assert classify_exception(ConnectError("synthetic pre-send name")) is Disposition.FAIL_FAST
