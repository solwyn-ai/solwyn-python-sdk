"""Anthropic server-side compaction usage: ``usage.iterations`` extraction.

Anthropic bills compaction but reports it ONLY in ``usage.iterations`` entries
of type ``compaction``; the top-level usage fields exclude it. The adapter adds
the normalized compaction sums into the parent buckets and reports them as the
``compaction_*`` subsets. Fixtures are SimpleNamespace / dict shapes; the real
beta pydantic types are exercised in ``test_real_sdk_anthropic_compaction.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._constants import ORDINARY_TOKEN_COUNT_MAX
from solwyn._token_details import TokenDetails
from solwyn.providers.anthropic import AnthropicAdapter, AnthropicStreamAccumulator

pytestmark = pytest.mark.unit


def _iteration(
    kind: str,
    *,
    input_tokens: object = 0,
    output_tokens: object = 0,
    cache_read: object = 0,
    cache_creation: object = 0,
    breakdown: tuple[object, object] | None = None,
) -> SimpleNamespace:
    fields: dict[str, object] = {
        "type": kind,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        "cache_creation": None,
    }
    if breakdown is not None:
        fields["cache_creation"] = SimpleNamespace(
            ephemeral_5m_input_tokens=breakdown[0],
            ephemeral_1h_input_tokens=breakdown[1],
        )
    return SimpleNamespace(**fields)


def _usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
    iterations: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        iterations=iterations,
    )


def _extract(usage: object) -> TokenDetails:
    return AnthropicAdapter().extract_usage(SimpleNamespace(usage=usage))


class TestCompactionExtraction:
    def test_threshold_mode_adds_compaction_iteration_to_parents(self) -> None:
        # Docs example: top-level excludes the compaction iteration.
        details = _extract(
            _usage(
                input_tokens=23_000,
                output_tokens=1_000,
                iterations=[
                    _iteration("compaction", input_tokens=180_000, output_tokens=3_500),
                    _iteration("message", input_tokens=23_000, output_tokens=1_000),
                ],
            )
        )
        assert details.input_tokens == 203_000
        assert details.output_tokens == 4_500
        assert details.compaction_input_tokens == 180_000
        assert details.compaction_output_tokens == 3_500
        assert details.compaction_cached_input_tokens == 0

    def test_on_demand_mode_with_zero_top_level_usage(self) -> None:
        details = _extract(
            _usage(
                iterations=[_iteration("compaction", input_tokens=144, output_tokens=276)],
            )
        )
        assert (details.input_tokens, details.output_tokens) == (144, 276)
        assert (details.compaction_input_tokens, details.compaction_output_tokens) == (144, 276)

    def test_cache_split_normalizes_compaction_like_the_top_level(self) -> None:
        details = _extract(
            _usage(
                input_tokens=500,
                cache_read=20_000,
                cache_creation=2_500,
                output_tokens=1_000,
                iterations=[
                    _iteration(
                        "compaction", input_tokens=2_000, cache_read=178_000, output_tokens=3_500
                    ),
                    _iteration(
                        "message",
                        input_tokens=500,
                        cache_read=20_000,
                        cache_creation=2_500,
                        output_tokens=1_000,
                    ),
                ],
            )
        )
        assert details.input_tokens == 2_500 + 198_000 + 2_500 == 203_000
        assert details.cached_input_tokens == 198_000
        assert details.cache_creation_5m_tokens == 2_500
        assert details.cache_creation_1h_tokens == 0
        assert details.output_tokens == 4_500
        assert details.compaction_input_tokens == 180_000
        assert details.compaction_cached_input_tokens == 178_000
        assert details.compaction_output_tokens == 3_500

    def test_compaction_cache_writes_use_ttl_breakdown_when_present(self) -> None:
        details = _extract(
            _usage(
                input_tokens=10,
                iterations=[
                    _iteration(
                        "compaction",
                        input_tokens=100,
                        cache_creation=999,  # ignored: the breakdown wins
                        breakdown=(40, 60),
                        output_tokens=5,
                    )
                ],
            )
        )
        assert details.compaction_input_tokens == 200
        assert details.compaction_cache_creation_5m_tokens == 40
        assert details.compaction_cache_creation_1h_tokens == 60
        assert details.cache_creation_5m_tokens == 40
        assert details.cache_creation_1h_tokens == 60
        assert details.input_tokens == 210

    def test_compaction_aggregate_cache_write_falls_back_to_5m(self) -> None:
        details = _extract(
            _usage(iterations=[_iteration("compaction", input_tokens=1, cache_creation=70)])
        )
        assert details.compaction_cache_creation_5m_tokens == 70
        assert details.cache_creation_5m_tokens == 70
        assert details.compaction_input_tokens == 71

    def test_multiple_compaction_entries_are_summed(self) -> None:
        details = _extract(
            _usage(
                iterations=[
                    _iteration("compaction", input_tokens=100, output_tokens=10),
                    _iteration("message", input_tokens=7, output_tokens=7),
                    _iteration("compaction", input_tokens=200, output_tokens=20),
                ],
            )
        )
        assert details.compaction_input_tokens == 300
        assert details.compaction_output_tokens == 30

    @pytest.mark.parametrize("kind", ["message", "advisor_message", "fallback_message", None])
    def test_non_compaction_iterations_are_ignored(self, kind: str | None) -> None:
        details = _extract(
            _usage(
                input_tokens=10,
                output_tokens=5,
                iterations=[_iteration(str(kind), input_tokens=9_999, output_tokens=9_999)],
            )
        )
        assert details == TokenDetails(input_tokens=10, output_tokens=5)

    def test_plain_dict_iterations_from_untyped_extra(self) -> None:
        # extra_body route: ``iterations`` is an untyped extra of plain dicts.
        details = _extract(
            _usage(
                input_tokens=10,
                output_tokens=5,
                iterations=[
                    {
                        "type": "compaction",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 0,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 3,
                            "ephemeral_1h_input_tokens": 4,
                        },
                    }
                ],
            )
        )
        assert details.compaction_input_tokens == 157
        assert details.compaction_cached_input_tokens == 50
        assert details.compaction_cache_creation_5m_tokens == 3
        assert details.compaction_cache_creation_1h_tokens == 4
        assert details.input_tokens == 167
        assert details.output_tokens == 25

    @pytest.mark.parametrize(
        "bad_entry",
        [
            _iteration("compaction", input_tokens=-1, output_tokens=5),
            _iteration("compaction", input_tokens=True, output_tokens=5),
            _iteration("compaction", input_tokens="100", output_tokens=5),
            _iteration("compaction", input_tokens=1.5, output_tokens=5),
            _iteration("compaction", input_tokens=5, output_tokens=-3),
            _iteration("compaction", input_tokens=5, cache_read=False),
            _iteration("compaction", input_tokens=5, breakdown=(-1, 0)),
            _iteration("compaction", input_tokens=5, breakdown=(0, "x")),
            {"type": "compaction", "input_tokens": None, "output_tokens": object()},
            "compaction",
            None,
            42,
        ],
    )
    def test_malformed_entries_are_skipped_never_raise(self, bad_entry: object) -> None:
        details = _extract(
            _usage(
                input_tokens=10,
                output_tokens=5,
                iterations=[bad_entry, _iteration("compaction", input_tokens=7, output_tokens=3)],
            )
        )
        assert details.input_tokens == 17
        assert details.output_tokens == 8
        assert details.compaction_input_tokens == 7
        assert details.compaction_output_tokens == 3

    def test_missing_optional_counts_read_as_zero(self) -> None:
        details = _extract(
            _usage(iterations=[{"type": "compaction", "input_tokens": 12, "output_tokens": 4}])
        )
        assert (details.compaction_input_tokens, details.compaction_output_tokens) == (12, 4)

    @pytest.mark.parametrize("iterations", [None, [], "not-a-list", {"type": "compaction"}])
    def test_absent_or_non_list_iterations_leave_wire_unchanged(self, iterations: object) -> None:
        fields = {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 5,
        }
        baseline = _extract(SimpleNamespace(**fields))  # pre-iterations shape
        details = _extract(SimpleNamespace(**fields, iterations=iterations))
        assert details == baseline
        assert details.model_dump_json() == baseline.model_dump_json()
        assert not any(key.startswith("compaction_") for key in details.model_dump())

    def test_usage_without_iterations_attribute_is_unchanged(self) -> None:
        usage = SimpleNamespace(input_tokens=3, output_tokens=4)
        assert _extract(usage) == TokenDetails(input_tokens=3, output_tokens=4)

    def test_oversized_compaction_is_clamped_so_validation_never_fails(self) -> None:
        huge = ORDINARY_TOKEN_COUNT_MAX + 5
        details = _extract(
            _usage(
                iterations=[
                    _iteration(
                        "compaction",
                        input_tokens=huge,
                        cache_read=huge,
                        breakdown=(huge, huge),
                        output_tokens=huge,
                    )
                ],
            )
        )
        assert details.input_tokens == 4 * huge
        assert details.compaction_input_tokens == ORDINARY_TOKEN_COUNT_MAX
        assert details.compaction_cached_input_tokens == ORDINARY_TOKEN_COUNT_MAX
        assert details.compaction_cache_creation_5m_tokens == 0
        assert details.compaction_cache_creation_1h_tokens == 0
        assert details.compaction_output_tokens == ORDINARY_TOKEN_COUNT_MAX


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _message_start(usage: object) -> SimpleNamespace:
    return SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage))


def _message_delta(**usage: object) -> SimpleNamespace:
    return SimpleNamespace(type="message_delta", usage=SimpleNamespace(**usage))


def _stream(*events: object) -> TokenDetails:
    acc = AnthropicStreamAccumulator()
    for event in events:
        acc.observe(event)
    return acc.finalize()


class TestCompactionStreaming:
    def test_iterations_from_last_message_delta_replace_not_add(self) -> None:
        details = _stream(
            _message_start(_usage(input_tokens=23_000, output_tokens=1)),
            _message_delta(
                output_tokens=500,
                iterations=[_iteration("compaction", input_tokens=90_000, output_tokens=1_000)],
            ),
            # Cumulative: the last delta carries the full list again.
            _message_delta(
                output_tokens=1_000,
                iterations=[
                    _iteration("compaction", input_tokens=180_000, output_tokens=3_500),
                    _iteration("message", input_tokens=23_000, output_tokens=1_000),
                ],
            ),
        )
        assert details.input_tokens == 203_000
        assert details.output_tokens == 4_500
        assert details.compaction_input_tokens == 180_000
        assert details.compaction_output_tokens == 3_500

    def test_delta_without_iterations_keeps_the_previous_delta_list(self) -> None:
        details = _stream(
            _message_start(_usage(input_tokens=10)),
            _message_delta(
                output_tokens=1,
                iterations=[_iteration("compaction", input_tokens=50, output_tokens=5)],
            ),
            _message_delta(output_tokens=2, iterations=None),
        )
        assert details.compaction_input_tokens == 50
        assert details.output_tokens == 7

    def test_falls_back_to_message_start_iterations(self) -> None:
        details = _stream(
            _message_start(
                _usage(
                    input_tokens=0,
                    iterations=[_iteration("compaction", input_tokens=144, output_tokens=276)],
                )
            ),
            _message_delta(output_tokens=0),
        )
        assert (details.input_tokens, details.output_tokens) == (144, 276)
        assert details.compaction_input_tokens == 144

    def test_cumulative_input_counts_on_message_delta_take_the_max(self) -> None:
        details = _stream(
            _message_start(_usage(input_tokens=100, cache_read=1_000)),
            _message_delta(
                output_tokens=50,
                input_tokens=400,
                cache_read_input_tokens=3_000,
                cache_creation_input_tokens=200,
            ),
        )
        assert details.input_tokens == 400 + 3_000 + 200
        assert details.cached_input_tokens == 3_000
        assert details.cache_creation_5m_tokens == 200
        assert details.output_tokens == 50

    @pytest.mark.parametrize("value", [0, None, -5, True, "7"])
    def test_zero_or_malformed_delta_counts_never_lower_message_start(self, value: object) -> None:
        start = SimpleNamespace(
            input_tokens=100,
            output_tokens=1,
            cache_read_input_tokens=1_000,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=30, ephemeral_1h_input_tokens=70
            ),
        )
        details = _stream(
            _message_start(start),
            _message_delta(
                output_tokens=9,
                input_tokens=value,
                cache_read_input_tokens=value,
                cache_creation_input_tokens=value,
            ),
        )
        assert details == TokenDetails(
            input_tokens=1_200,
            output_tokens=9,
            cached_input_tokens=1_000,
            cache_creation_5m_tokens=30,
            cache_creation_1h_tokens=70,
        )

    def test_delta_cache_creation_growth_goes_to_5m_preserving_start_breakdown(self) -> None:
        start = SimpleNamespace(
            input_tokens=10,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=30, ephemeral_1h_input_tokens=70
            ),
        )
        details = _stream(
            _message_start(start),
            _message_delta(output_tokens=1, cache_creation_input_tokens=150),
        )
        assert details.cache_creation_5m_tokens == 80
        assert details.cache_creation_1h_tokens == 70
        assert details.input_tokens == 160

    def test_stream_without_iterations_matches_prior_output(self) -> None:
        details = _stream(
            _message_start(_usage(input_tokens=25, cache_read=5)),
            _message_delta(output_tokens=11),
        )
        assert details == TokenDetails(input_tokens=30, output_tokens=11, cached_input_tokens=5)
        assert not any(key.startswith("compaction_") for key in details.model_dump())

    def test_stream_dict_iterations_are_parsed(self) -> None:
        details = _stream(
            _message_start(_usage(input_tokens=1)),
            _message_delta(
                output_tokens=1,
                iterations=[{"type": "compaction", "input_tokens": 8, "output_tokens": 2}],
            ),
        )
        assert (details.compaction_input_tokens, details.compaction_output_tokens) == (8, 2)


class TestBetaMessagesDispatchSeam:
    def test_prepare_beta_messages_call_selects_beta_create_and_passes_kwargs(self) -> None:
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        kwargs: dict[str, Any] = {
            "model": "claude-sonnet-4-6",
            "betas": ["compact-2026-01-12"],
            "context_management": {"edits": [{"type": "compact_20260112"}]},
        }
        method, prepared = AnthropicAdapter().prepare_beta_messages_call(
            client, kwargs, is_streaming=False
        )
        assert method is create
        assert prepared == kwargs
        assert prepared is not kwargs

    def test_prepare_beta_messages_call_streaming_sets_stream_without_mutation(self) -> None:
        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=dict)))
        kwargs: dict[str, Any] = {"model": "claude-sonnet-4-6", "compaction": {"type": "summarize"}}
        _, prepared = AnthropicAdapter().prepare_beta_messages_call(
            client, kwargs, is_streaming=True
        )
        assert prepared["stream"] is True
        assert prepared["compaction"] == {"type": "summarize"}
        assert "stream" not in kwargs
