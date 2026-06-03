"""call_id spend-reconciliation join key + possibly_succeeded abort flag (§8.4).

The Cloud API joins a served-provider MetadataEvent to its /budgets/confirm by
``call_id`` (one uuid per intercepted call), so cache-hit / abandoned-stream
spend can be reconciled and deduped exactly once. These tests prove the SDK
threads the SAME call_id into the success metadata event AND the confirm, and
that ``possibly_succeeded=True`` is set ONLY on a correctly-not-failed-over
post-send-ambiguous abort.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._types import CallStatus
from solwyn.client import Solwyn


class APITimeoutError(Exception):
    """Local stand-in for a provider read-timeout — classifies POST_SEND_AMBIGUOUS."""


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    return client


def _openai_response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok from gpt", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _allow_budget(reservation_id: str | None = "res_123") -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=reservation_id,
        project_id=None,
        budget_limit=100.0,
        current_usage=0.0,
        mode=SimpleNamespace(value="alert_only"),
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
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.mark.unit
class TestCallIdJoinKey:
    """The success metadata event and its confirm share one call_id."""

    def test_success_event_and_confirm_share_call_id(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-4o")

        confirm_spy = MagicMock()
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            patch.object(solwyn._budget, "confirm_cost", confirm_spy),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # The success metadata event carries a call_id…
        events = [c.args[0] for c in solwyn._reporter.report.call_args_list]
        success = [e for e in events if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        event_call_id = success[0].call_id
        assert event_call_id

        # …and the confirm was threaded the SAME call_id (the join key).
        confirm_spy.assert_called_once()
        assert confirm_spy.call_args.kwargs["call_id"] == event_call_id

        _close(solwyn)

    def test_call_id_unique_across_calls(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-4o")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        events = [c.args[0] for c in solwyn._reporter.report.call_args_list]
        ids = [e.call_id for e in events if e.status is CallStatus.SUCCESS]
        assert len(ids) == 2
        assert ids[0] != ids[1]

        _close(solwyn)


@pytest.mark.unit
class TestCacheHitReconciliation:
    """Budget-cache hit: confirm is skipped; the metadata event carries spend (§8.4)."""

    def test_cache_hit_skips_confirm_but_metadata_event_reconciles(self) -> None:
        # On a budget-cache hit the allow result has reservation_id=None, so the
        # SDK correctly SKIPS /budgets/confirm. The spec relies on the served-
        # provider SUCCESS MetadataEvent being sufficient for the Cloud API to
        # reconcile the cache-hit spend: it must carry the call_id join key, the
        # token_details, AND the served provider — the confirm-free path.
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-4o")

        confirm_spy = MagicMock()
        # reservation_id=None mimics a budget-cache hit (§8.4).
        with (
            patch.object(
                solwyn._budget, "check_budget", return_value=_allow_budget(reservation_id=None)
            ),
            patch.object(solwyn._budget, "confirm_cost", confirm_spy),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # (a) confirm_cost is NOT called on a cache hit.
        confirm_spy.assert_not_called()

        # (b) the SUCCESS metadata event ALONE carries everything the Cloud API
        #     needs to reconcile: a non-None call_id, token_details, and the
        #     served provider.
        events = [c.args[0] for c in solwyn._reporter.report.call_args_list]
        success = [e for e in events if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        ev = success[0]
        assert ev.call_id is not None and ev.call_id  # join key for reconciliation
        assert ev.token_details is not None  # served-provider spend signal
        assert ev.token_details.input_tokens == 10
        assert ev.token_details.output_tokens == 5
        assert ev.provider.value == "openai"  # served provider

        _close(solwyn)


@pytest.mark.unit
class TestPossiblySucceededAbortFlag:
    """possibly_succeeded=True only on the post-send-ambiguous abort path (§8.3)."""

    def test_post_send_ambiguous_abort_sets_possibly_succeeded(self) -> None:
        openai = _openai_client()
        original = APITimeoutError("read timed out")
        openai.chat.completions.create.side_effect = original
        # No fallback: safe default re-raises the ORIGINAL post-send-ambiguous exc.
        solwyn = _make_solwyn(openai, model="gpt-4o")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(APITimeoutError),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        events = [c.args[0] for c in solwyn._reporter.report.call_args_list]
        errors = [e for e in events if e.status is CallStatus.ERROR]
        assert len(errors) == 1
        # The abort event flags possibly_succeeded for Cloud-API reconciliation…
        assert errors[0].possibly_succeeded is True
        # …and still carries the per-call join key.
        assert errors[0].call_id

        _close(solwyn)

    def test_success_event_leaves_possibly_succeeded_none(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(openai, model="gpt-4o")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        events = [c.args[0] for c in solwyn._reporter.report.call_args_list]
        success = [e for e in events if e.status is CallStatus.SUCCESS]
        assert len(success) == 1
        # Non-abort events leave possibly_succeeded None (None-skipped on the wire).
        assert success[0].possibly_succeeded is None
        assert "possibly_succeeded" not in success[0].model_dump()

        _close(solwyn)
