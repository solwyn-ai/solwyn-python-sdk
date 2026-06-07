"""Cross-provider failover with REAL translation wired into dispatch.

The candidate walk no longer uses a cross-provider PASSTHROUGH; it now applies
the translation contract: on a real cross-provider hop the request is
reshaped via ``to_canonical``/``from_canonical`` BEFORE the network call, and the
served response is reshaped back into the caller's native dialect via
``normalize_response``. These tests exercise that seam end-to-end through
``Solwyn._intercepted_call`` with faked provider clients (no real SDKs importable).

Covered:
  * a fully-resolved tool exchange crosses OpenAI->Anthropic and the Anthropic
    call receives wrapped tool declarations + tool_result envelopes (id remap,
    JSON-object args);
  * an UNTRANSLATABLE feature that must cross providers raises
    UntranslatableRequestError, ABORTS the whole chain (fallback NOT attempted),
    and the error carries structural-only labels (no value leak into str(exc));
  * response normalization: an OpenAI-dialect call served by Anthropic exposes
    ``.choices[0].message.content`` (caller native path) with a normalized
    finish_reason;
  * the native happy path and same-provider model swap pay ZERO translation
    cost (``to_canonical`` is never called when the primary serves).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn.client import Solwyn
from solwyn.exceptions import UntranslatableModelError, UntranslatableRequestError
from solwyn.providers import _translation


class _Status(Exception):
    """Duck-typed transport error carrying an HTTP ``status_code``.

    ``status_code=429`` classifies FAILOVER (advance the chain) so the primary
    fails over to the cross-provider Anthropic candidate where translation runs.
    """

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


def _openai_text_response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok from gpt", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _anthropic_text_response(text: str = "ok from claude") -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        model="claude-3-5-sonnet",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _anthropic_length_response() -> SimpleNamespace:
    # stop_reason=max_tokens normalizes to canonical "length" -> OpenAI "length".
    block = SimpleNamespace(type="text", text="truncated")
    return SimpleNamespace(
        content=[block],
        stop_reason="max_tokens",
        model="claude-3-5-sonnet",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _anthropic_tool_response() -> SimpleNamespace:
    block = SimpleNamespace(
        type="tool_use", id="toolu_42", name="get_weather", input={"city": "paris"}
    )
    return SimpleNamespace(
        content=[block],
        stop_reason="tool_use",
        model="claude-3-5-sonnet",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _allow_budget() -> SimpleNamespace:
    return SimpleNamespace(allowed=True, reservation_id=None, price_hints=None)


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

# A fully-resolved tool exchange in the OpenAI dialect: tool_call issued AND its
# tool result returned (the stateless/resolvable case the subset supports).
_TOOL_REQUEST = {
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": "weather in paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "sunny, 21C"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}


# ── 1. tool exchange crosses OpenAI -> Anthropic ─────────────────────────


@pytest.mark.unit
class TestToolExchangeFailover:
    def test_target_native_default_params_do_not_enter_source_translation(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_text_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[
                (
                    anthropic,
                    "claude-3-5-sonnet",
                    {"max_tokens": 256, "top_k": 40},
                )
            ],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result.choices[0].message.content == "ok from claude"
        anthropic.messages.create.assert_called_once()
        kwargs = anthropic.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 256
        assert kwargs["top_k"] == 40

        _close(solwyn)

    def test_resolved_tool_exchange_translates_to_anthropic_native(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_tool_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_TOOL_REQUEST)

        anthropic.messages.create.assert_called_once()
        kwargs = anthropic.messages.create.call_args.kwargs

        # Tool declaration wrapped in Anthropic-native form (input_schema, no
        # OpenAI {type:function, function:{...}} envelope).
        assert kwargs["tools"] == [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        assert kwargs["tool_choice"] == {"type": "auto"}

        # tool_use block: id preserved (remap is consistent), args are a JSON
        # OBJECT (Anthropic dialect), NOT the OpenAI JSON-string form.
        assistant = kwargs["messages"][1]
        assert assistant["role"] == "assistant"
        tool_use = assistant["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["id"] == "call_abc"
        assert tool_use["name"] == "get_weather"
        assert tool_use["input"] == {"city": "paris"}

        # tool_result envelope keyed by the same tool_use_id (id remap).
        tool_turn = kwargs["messages"][2]
        assert tool_turn["role"] == "user"
        tool_result = tool_turn["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "call_abc"
        assert tool_result["content"] == "sunny, 21C"

        # The served Anthropic tool_use response is normalized back to the
        # OpenAI dialect the caller wrote.
        call = result.choices[0].message.tool_calls[0]
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"city": "paris"}
        assert result.choices[0].finish_reason == "tool_calls"

        _close(solwyn)


# ── 2. untranslatable feature ABORTS the chain ───────────────────────────


@pytest.mark.unit
class TestUntranslatableAbortsChain:
    @pytest.mark.parametrize(
        ("extra_kwargs", "expected_feature"),
        [
            ({"response_format": {"type": "json_object"}}, "response_format"),
            ({"seed": 7}, "seed"),
            ({"temperature": 1.5}, "temperature>1.0"),
        ],
    )
    def test_untranslatable_feature_aborts_whole_chain(
        self, extra_kwargs: dict[str, object], expected_feature: str
    ) -> None:
        # The primary 429s (FAILOVER), so the walk advances to the cross-provider
        # Anthropic candidate, where translation runs and RAISES before any
        # Anthropic network call — the whole chain aborts.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_text_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        sentinel = "SUPER_SECRET_PROMPT_abort"
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": sentinel}],
            **extra_kwargs,
        }

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableRequestError) as exc_info,
        ):
            solwyn.chat.completions.create(**request)

        exc = exc_info.value
        # Structural label only — the offending feature name, never its value.
        assert exc.feature == expected_feature
        assert exc.source == "openai"
        # No prompt content / offending value leaks into ANY string surface.
        assert sentinel not in str(exc)
        assert sentinel not in repr(exc)
        if extra_kwargs.get("response_format"):
            assert "json_object" not in str(exc)
        if "temperature" in extra_kwargs:
            assert "1.5" not in str(exc)
        # No provider exception text leaked via the chained-exception machinery.
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True

        # The fallback provider was NOT attempted — the chain aborted.
        anthropic.messages.create.assert_not_called()

        _close(solwyn)

    def test_dangling_tool_call_aborts_chain(self) -> None:
        # A tool_call still awaiting its result cannot be safely re-homed
        # -> RAISE on the first cross-provider hop.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_text_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        dangling = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_unresolved",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                # NOTE: no role:tool result resolving call_unresolved.
            ],
        }

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableRequestError) as exc_info,
        ):
            solwyn.chat.completions.create(**dangling)

        assert exc_info.value.feature == "dangling_tool_call"
        anthropic.messages.create.assert_not_called()

        _close(solwyn)

    def test_proprietary_tool_aborts_chain(self) -> None:
        # A non-function (proprietary) tool has no cross-provider meaning.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_text_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [{"type": "web_search_preview"}],
        }

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableRequestError) as exc_info,
        ):
            solwyn.chat.completions.create(**request)

        # Structural label names the proprietary tool type, never content.
        assert exc_info.value.feature == "openai.web_search_preview"
        anthropic.messages.create.assert_not_called()

        _close(solwyn)


# ── fix [G]: empty target model on a cross-provider hop -> abort up front ──


@pytest.mark.unit
class TestEmptyModelAbortsCrossProviderHop:
    def test_cross_provider_hop_with_empty_model_raises_untranslatable_model(self) -> None:
        # Missing model -> UntranslatableModelError up front. A
        # cross-provider fallback whose entry model is empty would otherwise send
        # an empty model to a healthy provider (a 400). The guard fires BEFORE any
        # translation/dispatch, aborting the chain — the fallback is NEVER called.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_text_response()

        # Empty model on the cross-provider fallback entry.
        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "", {"max_tokens": 256})],
        )
        anthropic_cb = solwyn._get_circuit_breaker("anthropic")
        before_state = anthropic_cb.get_state()

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableModelError) as exc_info,
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Structural error names the target provider + the (empty) model only.
        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.model == ""
        # The fallback provider was NEVER dispatched (aborted before the network).
        anthropic.messages.create.assert_not_called()
        # No breaker mutation on the target — a structural error, not a health signal.
        after_state = anthropic_cb.get_state()
        assert after_state.state == before_state.state
        assert after_state.failure_count == before_state.failure_count
        assert after_state.success_count == before_state.success_count

        _close(solwyn)


# ── 3. response normalization on the cross-provider hop ──────────────────


@pytest.mark.unit
class TestResponseNormalization:
    def test_anthropic_served_openai_caller_native_access_path(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_text_response("hello from claude")

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # The caller wrote OpenAI-dialect code; the normalized object exposes the
        # native access path with the Anthropic text.
        assert result.choices[0].message.content == "hello from claude"
        assert result.choices[0].message.role == "assistant"
        # end_turn normalizes through canonical "stop" -> OpenAI "stop".
        assert result.choices[0].finish_reason == "stop"

        _close(solwyn)

    def test_finish_reason_length_normalizes(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_length_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # Anthropic max_tokens -> canonical "length" -> OpenAI "length".
        assert result.choices[0].finish_reason == "length"

        _close(solwyn)


# ── 4. native happy path / same-provider swap pay ZERO translation cost ──


# ── 5. cross-provider STREAMING (text streams; tools fail loud) ──────────


@pytest.mark.unit
class TestCrossProviderStreamingFailsLoud:
    def test_cross_provider_text_streaming_now_fails_over(self) -> None:
        # The text fail-loud is now lifted: a PLAIN-TEXT cross-provider streaming
        # hop now fails over and the served (Anthropic) stream is normalized to
        # the caller's OpenAI dialect, served chunk-by-chunk.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter(
            [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="hi from claude"),
                )
            ]
        )

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        # The Anthropic fallback served and its text delta surfaces as an
        # OpenAI-dialect chunk (choices[0].delta.content).
        anthropic.messages.create.assert_called_once()
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "hi from claude"

        _close(solwyn)

    def test_cross_provider_tool_streaming_still_fails_loud(self) -> None:
        # The fail-loud is kept for the TOOL case: tool-call deltas are out of
        # the v1 streaming subset, so a tool-using cross-provider streaming hop
        # raises cross_provider_tool_stream BEFORE dispatch.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = _anthropic_text_response()

        solwyn = _make_solwyn(
            openai,
            model="gpt-4o",
            fallback=[(anthropic, "claude-3-5-sonnet", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableRequestError) as exc_info,
        ):
            solwyn.chat.completions.create(**request)

        assert exc_info.value.feature == "cross_provider_tool_stream"
        assert exc_info.value.source == "openai"
        # The foreign-dialect stream was never served.
        anthropic.messages.create.assert_not_called()

        _close(solwyn)

    def test_same_provider_streaming_model_swap_still_streams(self) -> None:
        # A same-provider streaming model swap must NOT be blocked — only the
        # cross-provider streaming hop fails loud.
        client = _openai_client()

        def _stream() -> object:
            return iter([SimpleNamespace(choices=[], usage=None)])

        client.chat.completions.create.side_effect = [_Status(429), _stream()]

        solwyn = _make_solwyn(client, model="gpt-4o", fallback=[(client, "gpt-4o-mini")])

        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**request)

        # The swap served (second call used the fallback model) and returned a
        # stream wrapper, not an UntranslatableRequestError.
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"
        assert result is not None

        _close(solwyn)


@pytest.mark.unit
class TestZeroTranslationOnNativePath:
    def test_native_happy_path_never_calls_to_canonical(self) -> None:
        client = _openai_client()
        native_resp = _openai_text_response()
        client.chat.completions.create.return_value = native_resp

        solwyn = _make_solwyn(client, model="gpt-4o")

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            patch.object(_translation, "to_canonical", wraps=_translation.to_canonical) as to_canon,
            patch.object(
                _translation, "normalize_response", wraps=_translation.normalize_response
            ) as normalize,
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        # The primary served natively: the raw response passes straight through
        # (identity) and the translator is never touched — zero overhead.
        assert result is native_resp
        to_canon.assert_not_called()
        normalize.assert_not_called()

        _close(solwyn)

    def test_same_provider_model_swap_never_calls_to_canonical(self) -> None:
        # Primary model 429s; the SAME client serves the gpt-4o-mini swap. A
        # same-provider hop only swaps the model string — no translation.
        client = _openai_client()
        success = _openai_text_response()
        client.chat.completions.create.side_effect = [_Status(429), success]

        solwyn = _make_solwyn(client, model="gpt-4o", fallback=[(client, "gpt-4o-mini")])

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            patch.object(_translation, "to_canonical", wraps=_translation.to_canonical) as to_canon,
            patch.object(
                _translation, "normalize_response", wraps=_translation.normalize_response
            ) as normalize,
        ):
            result = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert result is success
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"
        to_canon.assert_not_called()
        normalize.assert_not_called()

        _close(solwyn)
