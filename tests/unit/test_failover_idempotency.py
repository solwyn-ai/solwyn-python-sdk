"""Idempotency & double-spend safety.

Cross-provider failover is permitted ONLY when Solwyn can prove the request was
never durably accepted (pre-send: 429/529/connect-refused). A post-send
ambiguous failure (read timeout, 5xx-after-send) must NOT failover under the
default ``failover_idempotency="safe"`` policy — the ORIGINAL exception
re-raises unchanged so the caller's existing handlers fire.

``APITimeoutError`` is the canonical post-send-ambiguous case: classification
maps it by MRO name to POST_SEND_AMBIGUOUS without importing any provider SDK.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ProviderUnavailableError
from solwyn.providers._errors import Disposition, classify_exception


class APITimeoutError(Exception):
    """Local stand-in for the openai/anthropic read-timeout exception.

    classify_exception matches provider classes by MRO *name*, so naming this
    class ``APITimeoutError`` is sufficient to drive the POST_SEND_AMBIGUOUS
    branch — no real SDK import needed.

    NOTE the deliberate absence of a ``__cause__``: that is the CANONICAL read
    timeout, and a causeless wrapper stays POST_SEND_AMBIGUOUS. Use
    :func:`_wrapped_timeout` to build the pre-send-caused variant.
    """


def _wrapped_timeout(cause: Exception) -> APITimeoutError:
    """An ``APITimeoutError`` chaining ``cause``, as the real SDKs raise it.

    Both openai and anthropic catch the whole ``httpx.TimeoutException`` family
    and ``raise APITimeoutError(...) from err`` — so the pre-send
    ConnectTimeout/PoolTimeout cases are indistinguishable from a read timeout
    by class name alone. The chained cause is the only signal. Pinned against
    the real SDKs in tests/unit/test_real_sdk_error_wrapping.py.
    """
    exc = APITimeoutError(str(cause) or type(cause).__name__)
    exc.__cause__ = cause
    return exc


class _Status(Exception):
    """Duck-typed transport error carrying an HTTP ``status_code``."""

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


# ── client fakes ─────────────────────────────────────────────────────────


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    return client


def _anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"
    client.with_options.return_value = client
    return client


def _openai_response() -> SimpleNamespace:
    # Native OpenAI Chat Completions shape (duck-typed).
    message = SimpleNamespace(role="assistant", content="ok from gpt", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_response() -> SimpleNamespace:
    # Native Anthropic Messages shape (duck-typed): survives normalize_response
    # when it crosses back to an OpenAI-dialect caller.
    block = SimpleNamespace(type="text", text="ok from claude")
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _allow_budget() -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=None,
        project_id=VALID_PROJECT_ID,
        price_hints=None,
    )


def _make_solwyn(client: object, **overrides: object) -> Solwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    solwyn._reporter.report = MagicMock()
    return solwyn


def _close(solwyn: Solwyn) -> None:
    solwyn._reporter._http.close()
    solwyn._budget._http.close()


_PLAIN_REQUEST = {
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "hi"}],
}


# ── sanity: classification of the local stand-in ─────────────────────────


@pytest.mark.unit
def test_apitimeouterror_classifies_post_send_ambiguous() -> None:
    assert classify_exception(APITimeoutError("read timed out")) is Disposition.POST_SEND_AMBIGUOUS


@pytest.mark.unit
def test_status_429_classifies_failover() -> None:
    assert classify_exception(_Status(429)) is Disposition.FAILOVER


@pytest.mark.unit
@pytest.mark.parametrize(
    "cause",
    [httpx.ConnectTimeout("connect stalled"), httpx.PoolTimeout("pool exhausted")],
    ids=["connect", "pool"],
)
def test_wrapped_pre_send_timeout_classifies_failover(cause: Exception) -> None:
    # Same class NAME as the read-timeout case above, opposite disposition:
    # the connect/pool cause proves the request never reached the model.
    assert classify_exception(_wrapped_timeout(cause)) is Disposition.FAILOVER


@pytest.mark.unit
def test_wrapped_read_timeout_still_ambiguous() -> None:
    # The cause inspection must not weaken the post-send case it protects.
    assert (
        classify_exception(_wrapped_timeout(httpx.ReadTimeout("read stalled")))
        is Disposition.POST_SEND_AMBIGUOUS
    )


# ── the connect-slice regression (safe policy MUST still advance) ────────


@pytest.mark.unit
class TestWrappedConnectTimeoutAdvances:
    """PJ-8/R7 made the connect slice short and deadline-derived, so on a stalled
    primary it is the bound that fires FIRST. Classifying the resulting
    ``APITimeoutError`` by name alone stranded the chain: default-safe
    idempotency re-raised and the fallback was never attempted. These are the
    end-to-end controls for that regression."""

    def test_connect_slice_timeout_fails_over_under_safe_default(self) -> None:
        # Arrange: primary raises the SDK-shaped wrapper over a ConnectTimeout.
        # No idempotency override — this is the DEFAULT "safe" policy.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _wrapped_timeout(
            httpx.ConnectTimeout("connect slice expired")
        )
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        # Act
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Assert: the chain ADVANCED — a pre-send failure cannot double-spend.
        anthropic.messages.create.assert_called_once()
        assert result.choices[0].message.content == "ok from claude"

        _close(solwyn)

    def test_pool_timeout_fails_over_under_safe_default(self) -> None:
        # PoolTimeout is the other pre-send cause the connect slice produces:
        # the request never even got a connection out of the pool.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _wrapped_timeout(
            httpx.PoolTimeout("pool slice expired")
        )
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        anthropic.messages.create.assert_called_once()
        assert result.choices[0].message.content == "ok from claude"

        _close(solwyn)

    def test_read_timeout_still_reraises_under_safe_default(self) -> None:
        # The guard on the guard: the SAME wrapper class over a READ cause must
        # keep its post-send protection. If this ever goes green-by-failover the
        # cause inspection has been widened into a double-spend.
        openai = _openai_client()
        original = _wrapped_timeout(httpx.ReadTimeout("read stalled"))
        openai.chat.completions.create.side_effect = original
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(APITimeoutError) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value is original
        anthropic.messages.create.assert_not_called()

        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_connect_slice_timeout_fails_over_under_safe_default(self) -> None:
        # Async mirror: the walk is a separate code path, and the classification
        # seam it calls must be reached identically.
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(
            side_effect=_wrapped_timeout(httpx.ConnectTimeout("connect slice expired"))
        )
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(return_value=_anthropic_response())

        solwyn = AsyncSolwyn(
            openai,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        solwyn._reporter.report = MagicMock(spec=solwyn._reporter.report)

        with patch.object(
            solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            result = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        anthropic.messages.create.assert_awaited_once()
        assert result.choices[0].message.content == "ok from claude"

        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()


# ── safe (default) ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestSafeDefault:
    def test_apitimeout_does_not_failover_reraises_original(self) -> None:
        openai = _openai_client()
        original = APITimeoutError("read timed out")
        openai.chat.completions.create.side_effect = original
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(APITimeoutError) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # The EXACT original instance propagates; the fallback never ran.
        assert exc_info.value is original
        anthropic.messages.create.assert_not_called()

        _close(solwyn)

    def test_real_httpx_readtimeout_does_not_failover_reraises_original(self) -> None:
        openai = _openai_client()
        original = httpx.ReadTimeout("read timed out")
        openai.chat.completions.create.side_effect = original
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(httpx.ReadTimeout) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert exc_info.value is original
        anthropic.messages.create.assert_not_called()

        _close(solwyn)

    def test_429_does_failover(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create.return_value = anthropic_resp

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # A 429 is a provable pre-send rejection -> safe to cross providers.
        # The crossed response is normalized back to the OpenAI dialect.
        assert result is not anthropic_resp
        assert result.choices[0].message.content == "ok from claude"
        anthropic.messages.create.assert_called_once()

        _close(solwyn)


# ── always ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestAlwaysMode:
    def test_apitimeout_fails_over_when_always(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = APITimeoutError("read timed out")
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create.return_value = anthropic_resp

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            failover_idempotency="always",
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # always = caller asserts idempotency -> ambiguous failures DO cross.
        # The crossed response is normalized back to the OpenAI dialect.
        assert result is not anthropic_resp
        assert result.choices[0].message.content == "ok from claude"
        anthropic.messages.create.assert_called_once()

        _close(solwyn)

    def test_failed_over_psa_error_event_never_flags_possibly_succeeded(self) -> None:
        # Under failover_idempotency="always" a POST_SEND_AMBIGUOUS
        # primary failure DOES fail over to the fallback. Because the call failed
        # over, its primary ERROR event must carry possibly_succeeded None (absent
        # from model_dump) — never True — so reconciliation cannot double-count a
        # call that both failed over AND is flagged possibly-succeeded.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = APITimeoutError("read timed out")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            failover_idempotency="always",
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # The PSA failure DID fail over: the fallback served.
        anthropic.messages.create.assert_called_once()
        errors = [e for e in events if e.status.value == "error"]
        success = [e for e in events if e.status.value == "success"]
        assert len(errors) == 1  # primary PSA error event
        assert len(success) == 1  # fallback served
        # The failed-over primary error event NEVER flags possibly_succeeded.
        assert errors[0].possibly_succeeded is None
        assert "possibly_succeeded" not in errors[0].model_dump()

        _close(solwyn)

    def test_per_call_idempotent_failed_over_psa_error_event_clean(self) -> None:
        # Per-call variant: solwyn_idempotent=True escalates the
        # default "safe" policy to allow ambiguous failover. The PSA primary
        # error still fails over and its error event stays clean (None, absent).
        openai = _openai_client()
        openai.chat.completions.create.side_effect = APITimeoutError("read timed out")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        events: list = []
        solwyn._reporter.report = lambda e: events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(solwyn_idempotent=True, **_PLAIN_REQUEST)

        anthropic.messages.create.assert_called_once()
        errors = [e for e in events if e.status.value == "error"]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is None
        assert "possibly_succeeded" not in errors[0].model_dump()

        _close(solwyn)


# ── never ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestNeverMode:
    def test_429_does_not_cross_to_different_provider(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429, "rate limited")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            failover_idempotency="never",
        )

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status, match="rate limited"),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # never = breaker still tracks health, but NEVER cross providers.
        anthropic.messages.create.assert_not_called()

        _close(solwyn)


@pytest.mark.unit
class TestSameProviderOpenAITokenKeyRewrite:
    def test_o_series_model_swap_rewrites_max_tokens_default(self) -> None:
        client = _openai_client()
        success = _openai_response()
        client.chat.completions.create.side_effect = [_Status(429), success]

        solwyn = _make_solwyn(
            client,
            model="gpt-5.5",
            fallback=[(client, "o3-mini")],
            default_params={"max_tokens": 256},
            failover_idempotency="never",
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        swap_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert swap_kwargs["model"] == "o3-mini"
        assert swap_kwargs["max_completion_tokens"] == 256
        assert "max_tokens" not in swap_kwargs

        _close(solwyn)

    def test_same_provider_model_swap_still_occurs_when_never(self) -> None:
        # Two same-provider entries: never blocks cross-PROVIDER hops only, so
        # the in-provider model swap is still allowed.
        client = _openai_client()
        success = _openai_response()
        client.chat.completions.create.side_effect = [_Status(429), success]

        solwyn = _make_solwyn(
            client,
            model="gpt-5.5",
            fallback=[(client, "gpt-5.4-mini")],
            failover_idempotency="never",
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-5.4-mini"

        _close(solwyn)


# ── per-call override ────────────────────────────────────────────────────


@pytest.mark.unit
class TestPerCallOverride:
    def test_solwyn_idempotent_true_allows_ambiguous_failover_and_is_stripped(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = APITimeoutError("read timed out")
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create.return_value = anthropic_resp

        # Default policy is "safe"; the per-call flag escalates to ambiguous-OK.
        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(
                solwyn_idempotent=True,
                **_PLAIN_REQUEST,
            )

        # The crossed response is normalized back to the OpenAI dialect.
        assert result is not anthropic_resp
        assert result.choices[0].message.content == "ok from claude"
        anthropic.messages.create.assert_called_once()
        # solwyn_idempotent must be stripped before dispatch on BOTH hops.
        assert "solwyn_idempotent" not in openai.chat.completions.create.call_args.kwargs
        assert "solwyn_idempotent" not in anthropic.messages.create.call_args.kwargs

        _close(solwyn)


# ── same-provider breaker double-count guard ─────────────────────────────


@pytest.mark.unit
class TestSameProviderDoubleCountGuard:
    def test_two_same_provider_failures_count_breaker_once(self) -> None:
        # A chain of two same-provider entries that both 429 must record the
        # provider breaker failure only ONCE per logical call — otherwise the
        # breaker opens twice as fast as configured.
        client = _openai_client()
        client.chat.completions.create.side_effect = [_Status(429), _Status(429)]

        solwyn = _make_solwyn(client, model="gpt-5.5", fallback=[(client, "gpt-5.4-mini")])
        cb = solwyn._get_circuit_breaker("openai")
        assert cb.failure_threshold == 3  # default

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Both hops were attempted, but the breaker counted exactly one failure.
        assert client.chat.completions.create.call_count == 2
        assert cb.failure_count == 1
        assert cb.state.value == "closed"  # still below the threshold

        _close(solwyn)

    def test_breaker_opens_at_threshold_not_twice_as_fast(self) -> None:
        # Three logical calls, each with two same-provider failing entries.
        # With single-count semantics the breaker opens after exactly 3 calls
        # (== failure_threshold), not after 2 (which double-counting would give).
        client = _openai_client()
        client.chat.completions.create.side_effect = [_Status(429)] * 6

        solwyn = _make_solwyn(client, model="gpt-5.5", fallback=[(client, "gpt-5.4-mini")])
        cb = solwyn._get_circuit_breaker("openai")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            # Calls 1 and 2: breaker still CLOSED afterwards.
            for _ in range(2):
                with pytest.raises(_Status):
                    solwyn.chat.completions.create(**_PLAIN_REQUEST)
            assert cb.failure_count == 2
            assert cb.state.value == "closed"

            # Call 3 trips the threshold -> the chain finds no eligible
            # candidate and raises ProviderUnavailableError.
            with pytest.raises((ProviderUnavailableError, _Status)):
                solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert cb.state.value == "open"

        _close(solwyn)
