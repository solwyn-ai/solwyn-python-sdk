"""Idempotency & double-spend safety — the P1 DoD (spec §6.5).

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
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn.client import Solwyn
from solwyn.providers._errors import Disposition, classify_exception


class APITimeoutError(Exception):
    """Local stand-in for the openai/anthropic read-timeout exception.

    classify_exception matches provider classes by MRO *name*, so naming this
    class ``APITimeoutError`` is sufficient to drive the POST_SEND_AMBIGUOUS
    branch — no real SDK import needed.
    """


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


def _openai_response() -> MagicMock:
    resp = MagicMock()
    resp.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return resp


def _anthropic_response() -> MagicMock:
    resp = MagicMock()
    resp.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return resp


def _allow_budget() -> SimpleNamespace:
    return SimpleNamespace(allowed=True, reservation_id=None)


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
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
}


# ── sanity: classification of the local stand-in ─────────────────────────


@pytest.mark.unit
def test_apitimeouterror_classifies_post_send_ambiguous() -> None:
    assert classify_exception(APITimeoutError("read timed out")) is Disposition.POST_SEND_AMBIGUOUS


@pytest.mark.unit
def test_status_429_classifies_failover() -> None:
    assert classify_exception(_Status(429)) is Disposition.FAILOVER


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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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

    def test_429_does_failover(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic_resp = _anthropic_response()
        anthropic.messages.create.return_value = anthropic_resp

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # A 429 is a provable pre-send rejection -> safe to cross providers.
        assert result is anthropic_resp
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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
            failover_idempotency="always",
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # always = caller asserts idempotency -> ambiguous failures DO cross.
        assert result is anthropic_resp
        anthropic.messages.create.assert_called_once()

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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
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

    def test_same_provider_model_swap_still_occurs_when_never(self) -> None:
        # Two same-provider entries: never blocks cross-PROVIDER hops only, so
        # the in-provider model swap is still allowed.
        client = _openai_client()
        success = _openai_response()
        client.chat.completions.create.side_effect = [_Status(429), success]

        solwyn = _make_solwyn(
            client,
            model="gpt-4o",
            fallback=[(client, "gpt-4o-mini")],
            failover_idempotency="never",
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"

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
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(
                solwyn_idempotent=True,
                **_PLAIN_REQUEST,
            )

        assert result is anthropic_resp
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
        # breaker opens twice as fast as configured (§4.6).
        client = _openai_client()
        client.chat.completions.create.side_effect = [_Status(429), _Status(429)]

        solwyn = _make_solwyn(client, model="gpt-4o", fallback=[(client, "gpt-4o-mini")])
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

        solwyn = _make_solwyn(client, model="gpt-4o", fallback=[(client, "gpt-4o-mini")])
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
            from solwyn.exceptions import ProviderUnavailableError

            with pytest.raises((ProviderUnavailableError, _Status)):
                solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert cb.state.value == "open"

        _close(solwyn)
