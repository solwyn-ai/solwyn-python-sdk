"""Outage-durable folding for denied-call receipts.

Receipt folds are deliberately content-free and bounded.  They ride the normal
metadata queue only after a successful event-delivery cycle establishes that
the ingest plane has recovered.
"""

from __future__ import annotations

import dataclasses
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, call_uuid

from solwyn._lifecycle import blocking_exit_flush
from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetConfirmRequest,
    CallStatus,
    MediaUsage,
    MetadataEvent,
    ProviderName,
)
from solwyn.reporter import (
    AsyncMetadataReporter,
    MetadataReporter,
    _build_receipt_replay_events,
    _PendingEvent,
    _SendOutcome,
    _split_receipt_fold,
)

_URL = "https://api.test.solwyn.ai"
_WIRE_QUANTITY_MAX = 100_000_000


def _event(**overrides: object) -> MetadataEvent:
    values: dict[str, object] = {
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "input_tokens": 3,
        "output_tokens": 0,
        "latency_ms": 0.0,
        "status": CallStatus.BUDGET_DENIED,
        "is_model_fallback": False,
        "sdk_instance_id": "fold-test-instance",
        "timestamp": datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
        "call_id": call_uuid(f"fold-{len(str(overrides))}-{overrides.get('agent_run_id', '')}"),
        "agent_run_id": "run-fold",
        "deny_source": "server",
        "deny_reason": "structural-reason",
        "estimated_output_bound": 7,
    }
    values.update(overrides)
    return MetadataEvent(**values)


def _confirm(**overrides: object) -> BudgetConfirmRequest:
    values: dict[str, object] = {
        "reservation_id": "res_fold",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "call_id": call_uuid("fold-confirm"),
        "token_details": TokenDetails(input_tokens=1, output_tokens=0),
    }
    values.update(overrides)
    return BudgetConfirmRequest(**values)


def _ok_response() -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock(spec=httpx.Response.raise_for_status)
    response.json = MagicMock(
        spec=httpx.Response.json,
        return_value={"ingested": 1, "rejected": []},
    )
    return response


def _quiet_sync(**kwargs: object) -> MetadataReporter:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(_URL, VALID_API_KEY, **kwargs)
    reporter._thread.join(timeout=2.0)
    return reporter


def _only_fold(reporter: MetadataReporter | AsyncMetadataReporter):
    snapshot = reporter._receipt_fold_snapshot()
    assert len(snapshot) == 1
    return next(iter(snapshot.values()))


class _GateAfterNthOwnershipRelease:
    """Test lock that pauses one named worker after an ownership transaction."""

    def __init__(self, worker_name: str, release_number: int) -> None:
        self._lock = threading.Lock()
        self._worker_name = worker_name
        self._release_number = release_number
        self._worker_releases = 0
        self.released = threading.Event()
        self.allow_worker = threading.Event()

    def __enter__(self) -> _GateAfterNthOwnershipRelease:
        self._lock.acquire()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self._lock.release()
        if threading.current_thread().name != self._worker_name:
            return
        self._worker_releases += 1
        if self._worker_releases == self._release_number:
            self.released.set()
            assert self.allow_worker.wait(timeout=5.0)


@pytest.mark.unit
class TestReceiptFoldState:
    def test_same_pricing_identity_accumulates_counts_sums_and_timestamps(self) -> None:
        reporter = _quiet_sync()
        first_at = datetime(2026, 8, 17, 12, 1, tzinfo=UTC)
        last_at = first_at + timedelta(seconds=4)
        try:
            reporter._fold_or_count_event_drop(
                _event(
                    timestamp=last_at,
                    input_tokens=5,
                    estimated_output_bound=11,
                    receipt_aggregate_count=40,
                ),
                "retry_exhausted",
            )
            reporter._fold_or_count_event_drop(
                _event(
                    timestamp=first_at,
                    input_tokens=7,
                    estimated_output_bound=13,
                    receipt_aggregate_count=60,
                ),
                "retry_exhausted",
            )

            fold = _only_fold(reporter)
            assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (100, 12, 24)
            assert (fold.first_timestamp, fold.last_timestamp) == (first_at, last_at)
            assert (fold.model, fold.provider) == ("gpt-5.5", ProviderName.OPENAI)
            assert reporter.dropped_counts == {}
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_capacity_retains_256_keys_overflows_257th_and_updates_existing(self) -> None:
        reporter = _quiet_sync()
        try:
            for i in range(256):
                reporter._fold_or_count_event_drop(
                    _event(agent_run_id=f"run-{i}", input_tokens=1), "overflow"
                )
            reporter._fold_or_count_event_drop(
                _event(
                    agent_run_id="run-overflow",
                    input_tokens=2,
                    receipt_aggregate_count=7,
                ),
                "overflow",
            )
            reporter._fold_or_count_event_drop(
                _event(agent_run_id="run-0", input_tokens=1), "overflow"
            )

            snapshot = reporter._receipt_fold_snapshot()
            assert len(snapshot) == 256
            assert not any(key[0] == "run-overflow" for key in snapshot)
            run_zero = next(fold for key, fold in snapshot.items() if key[0] == "run-0")
            assert run_zero.count == 2
            assert run_zero.input_tokens == 2
            assert reporter.dropped_counts == {"event.receipt_fold_overflow": 7}
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_growing_prompt_runaway_folds_coarsely_instead_of_starving_the_table(
        self,
    ) -> None:
        # The flagship runaway: every call grows the prompt, so every receipt
        # has a distinct pricing basis. Before the per-run exact budget those
        # 300 receipts minted 256 single-receipt keys and refused the rest —
        # for this run AND every other run on the client.
        reporter = _quiet_sync()
        try:
            for index in range(300):
                reporter._fold_or_count_event_drop(
                    _event(
                        call_id=call_uuid(f"runaway-{index}"),
                        agent_run_id="runaway",
                        input_tokens=1_000 + index,
                    ),
                    "retry_exhausted",
                )

            snapshot = reporter._receipt_fold_snapshot()
            runaway = {key: fold for key, fold in snapshot.items() if key.run_id == "runaway"}
            exact = {
                key: fold
                for key, fold in runaway.items()
                if key.receipt_pricing_input_tokens is not None
            }
            coarse = {
                key: fold
                for key, fold in runaway.items()
                if key.receipt_pricing_input_tokens is None
            }
            # 32 exact keys, then ONE coarse aggregate for the remainder.
            assert len(exact) == 32
            assert len(coarse) == 1
            assert sum(fold.count for fold in runaway.values()) == 300
            assert next(iter(coarse.values())).count == 268
            # Nothing was refused, and the shared table stays mostly free.
            assert reporter.dropped_counts == {}
            assert len(snapshot) == 33

            # A second run still gets its own EXACT pricing basis.
            reporter._fold_or_count_event_drop(
                _event(
                    call_id=call_uuid("bystander"),
                    agent_run_id="bystander",
                    input_tokens=4_242,
                ),
                "retry_exhausted",
            )
            bystander = [
                key for key in reporter._receipt_fold_snapshot() if key.run_id == "bystander"
            ]
            assert len(bystander) == 1
            assert bystander[0].receipt_pricing_input_tokens == 4_242

            reporter._drain_receipt_folds_to_queue(final=True)
            replays = [pending.event for pending in reporter._queue]
            runaway_replays = [event for event in replays if event.agent_run_id == "runaway"]
            assert sum(event.receipt_aggregate_count or 0 for event in runaway_replays) == 300
            coarse_replays = [
                event for event in runaway_replays if event.receipt_pricing_input_tokens is None
            ]
            # A null pricing basis is how the server tells a coarse aggregate
            # (unpriceable per card) from an exactly priceable one.
            assert len(coarse_replays) == 1
            assert coarse_replays[0].receipt_aggregate_count == 268
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_denial_reason_and_period_partition_folds_and_survive_replay(self) -> None:
        reporter = _quiet_sync()
        try:
            reporter._fold_or_count_event_drop(
                _event(
                    call_id=call_uuid("reason-stop"),
                    deny_reason="run_stopped",
                    denied_by_period="agent_run",
                    velocity_flags=["monotonic_growth"],
                ),
                "retry_exhausted",
            )
            reporter._fold_or_count_event_drop(
                _event(
                    call_id=call_uuid("reason-stop-2"),
                    deny_reason="run_stopped",
                    denied_by_period="agent_run",
                    velocity_flags=["repeat_size"],
                ),
                "retry_exhausted",
            )
            reporter._fold_or_count_event_drop(
                _event(
                    call_id=call_uuid("reason-monthly"),
                    deny_reason="budget_exceeded",
                    denied_by_period="monthly",
                ),
                "retry_exhausted",
            )

            # Same run, same model, same size — the denial evidence differs, so
            # the aggregates must not merge.
            assert len(reporter._receipt_fold_snapshot()) == 2

            reporter._drain_receipt_folds_to_queue(final=True)
            replays = {
                (event.deny_reason, event.denied_by_period): event
                for event in (pending.event for pending in reporter._queue)
            }
            assert set(replays) == {("run_stopped", "agent_run"), ("budget_exceeded", "monthly")}
            stopped = replays[("run_stopped", "agent_run")]
            assert stopped.receipt_aggregate_count == 2
            # Flags are unioned, not keyed: they name rules, not pricing.
            assert stopped.velocity_flags == ["monotonic_growth", "repeat_size"]
            assert replays[("budget_exceeded", "monthly")].velocity_flags is None
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_sources_stay_distinct_and_empty_key_is_supported(self) -> None:
        reporter = _quiet_sync()
        try:
            reporter._fold_or_count_event_drop(_event(deny_source="server"), "overflow")
            reporter._fold_or_count_event_drop(_event(deny_source="local_velocity"), "overflow")
            reporter._fold_or_count_event_drop(
                _event(agent_run_id=None, deny_source=None), "overflow"
            )
            assert {(key[0], key[1]) for key in reporter._receipt_fold_snapshot()} == {
                ("run-fold", "server"),
                ("run-fold", "local_velocity"),
                ("", ""),
            }
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_pricing_selectors_partition_folds_and_media_quantities_sum_exactly(self) -> None:
        reporter = _quiet_sync()
        try:
            compatible = {
                "model": "gpt-image-2",
                "provider": ProviderName.OPENAI,
                "provider_region": None,
                "service_tier": "priority",
                "modality": "image",
            }
            reporter._fold_or_count_event_drop(
                _event(
                    **compatible,
                    input_tokens=11,
                    output_tokens=13,
                    media_usage=MediaUsage(
                        image_count=2,
                        generation_count=1,
                        resolution="1024x1024",
                        quality="hd",
                    ),
                ),
                "retry_exhausted",
            )
            reporter._fold_or_count_event_drop(
                _event(
                    **compatible,
                    input_tokens=11,
                    output_tokens=19,
                    media_usage=MediaUsage(
                        image_count=3,
                        generation_count=4,
                        resolution="1024x1024",
                        quality="hd",
                    ),
                ),
                "retry_exhausted",
            )
            for overrides in (
                {"model": "gpt-image-1"},
                {"provider": ProviderName.BEDROCK, "provider_region": "us-east-1"},
                {"provider": ProviderName.BEDROCK, "provider_region": "eu-west-1"},
                {"service_tier": "flex"},
                {
                    "media_usage": MediaUsage(
                        image_count=1,
                        resolution="512x512",
                        quality="hd",
                    )
                },
            ):
                reporter._fold_or_count_event_drop(
                    _event(**{**compatible, **overrides}),
                    "retry_exhausted",
                )

            assert len(reporter._receipt_fold_snapshot()) == 6
            reporter._drain_receipt_folds_to_queue(final=True)
            replays = [pending.event for pending in reporter._queue]
            aggregate = next(
                event
                for event in replays
                if event.model == "gpt-image-2"
                and event.provider is ProviderName.OPENAI
                and event.service_tier == "priority"
                and event.media_usage is not None
                and event.media_usage.resolution == "1024x1024"
            )
            assert (aggregate.input_tokens, aggregate.output_tokens) == (22, 32)
            assert aggregate.media_usage == MediaUsage(
                image_count=5,
                generation_count=5,
                resolution="1024x1024",
                quality="hd",
            )
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_split_of_an_invariant_violating_fold_overcounts_instead_of_raising(
        self,
    ) -> None:
        # take_for_cycle is a DESTRUCTIVE take: the snapshot is already gone by
        # the time these events are built, so a wire-validation raise here
        # would erase every folded receipt with zero drop accounting. The
        # invariant (chunk_count <= count, because each source receipt carries
        # at most one wire-max quantity) is what keeps counts >= 1; if it ever
        # broke, the clamp must keep the failure a conservative OVERcount.
        reporter = _quiet_sync()
        try:
            reporter._fold_or_count_event_drop(_event(input_tokens=1), "retry_exhausted")
            key, fold = next(iter(reporter._receipt_fold_snapshot().items()))
            violating = dataclasses.replace(fold, count=1, input_tokens=2 * _WIRE_QUANTITY_MAX)

            chunks = _split_receipt_fold(violating)
            assert len(chunks) == 2
            assert [chunk.count for chunk in chunks] == [1, 1]

            events = _build_receipt_replay_events(key, violating, "split-clamp")
            assert sum(event.receipt_aggregate_count or 0 for event in events) >= violating.count
            assert sum(event.input_tokens for event in events) == 2 * _WIRE_QUANTITY_MAX
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_media_optional_presence_is_preserved_for_core_compatible_replay(self) -> None:
        reporter = _quiet_sync()
        cases = {
            "image": MediaUsage(image_count=2, generation_count=1),
            "audio": MediaUsage(audio_seconds=2.5),
            "video": MediaUsage(video_seconds=3.5),
            "characters": MediaUsage(input_characters=40),
            "unknown": MediaUsage(),
        }
        try:
            for run_id, media_usage in cases.items():
                for suffix in ("a", "b"):
                    reporter._fold_or_count_event_drop(
                        _event(
                            call_id=call_uuid(f"media-presence-{run_id}-{suffix}"),
                            agent_run_id=run_id,
                            modality=(
                                "audio" if run_id in {"audio", "characters", "unknown"} else run_id
                            ),
                            media_usage=media_usage,
                        ),
                        "retry_exhausted",
                    )

            reporter._drain_receipt_folds_to_queue(final=True)
            by_run = {pending.event.agent_run_id: pending.event for pending in reporter._queue}
            assert by_run["image"].media_usage == MediaUsage(image_count=4, generation_count=2)
            assert by_run["audio"].media_usage == MediaUsage(audio_seconds=5.0)
            assert by_run["video"].media_usage == MediaUsage(video_seconds=7.0)
            assert by_run["characters"].media_usage == MediaUsage(input_characters=80)
            assert by_run["unknown"].media_usage == MediaUsage()

            for event in by_run.values():
                assert event.receipt_pricing_input_tokens == 3
                media_payload = event.model_dump(mode="json", exclude_none=True)["media_usage"]
                assert not ({"audio_seconds", "video_seconds"} <= media_payload.keys())
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_explicit_zero_media_quantity_does_not_fold_with_absent_quantity(self) -> None:
        reporter = _quiet_sync()
        try:
            for suffix in ("a", "b"):
                reporter._fold_or_count_event_drop(
                    _event(
                        call_id=call_uuid(f"explicit-zero-{suffix}"),
                        media_usage=MediaUsage(image_count=0),
                        modality="image",
                        receipt_aggregate_count=_WIRE_QUANTITY_MAX,
                    ),
                    "retry_exhausted",
                )
                reporter._fold_or_count_event_drop(
                    _event(
                        call_id=call_uuid(f"absent-zero-{suffix}"),
                        media_usage=MediaUsage(),
                        modality="image",
                        receipt_aggregate_count=_WIRE_QUANTITY_MAX,
                    ),
                    "retry_exhausted",
                )

            assert len(reporter._receipt_fold_snapshot()) == 2
            reporter._drain_receipt_folds_to_queue(final=True)
            image_counts = [
                pending.event.media_usage.image_count
                if pending.event.media_usage is not None
                else None
                for pending in reporter._queue
            ]
            assert sorted(image_counts, key=lambda value: value is not None) == [None, None, 0, 0]
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_original_input_token_basis_partitions_and_prices_aggregate_replays(self) -> None:
        reporter = _quiet_sync()
        try:
            for suffix in ("a", "b"):
                reporter._fold_or_count_event_drop(
                    _event(
                        call_id=call_uuid(f"gemini-short-{suffix}"),
                        model="gemini-2.5-pro",
                        provider=ProviderName.GOOGLE,
                        input_tokens=150_000,
                    ),
                    "retry_exhausted",
                )
            reporter._fold_or_count_event_drop(
                _event(
                    model="gemini-2.5-pro",
                    provider=ProviderName.GOOGLE,
                    input_tokens=250_000,
                ),
                "retry_exhausted",
            )
            for input_tokens in (272_000, 272_001):
                reporter._fold_or_count_event_drop(
                    _event(
                        model="gpt-5.6-sol",
                        provider=ProviderName.OPENAI,
                        input_tokens=input_tokens,
                    ),
                    "retry_exhausted",
                )

            assert len(reporter._receipt_fold_snapshot()) == 4
            reporter._drain_receipt_folds_to_queue(final=True)
            replays = [pending.event for pending in reporter._queue]
            gemini_short = next(
                event
                for event in replays
                if event.model == "gemini-2.5-pro" and event.receipt_pricing_input_tokens == 150_000
            )
            assert (gemini_short.input_tokens, gemini_short.receipt_aggregate_count) == (
                300_000,
                2,
            )
            assert {
                event.receipt_pricing_input_tokens
                for event in replays
                if event.model == "gemini-2.5-pro"
            } == {150_000, 250_000}
            assert {
                event.receipt_pricing_input_tokens
                for event in replays
                if event.model == "gpt-5.6-sol"
            } == {272_000, 272_001}
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_aggregate_replays_split_at_100m_and_refold_exactly(self) -> None:
        reporter = _quiet_sync()
        try:
            for suffix in ("a", "b", "c"):
                reporter._fold_or_count_event_drop(
                    _event(
                        call_id=call_uuid(f"100m-source-{suffix}"),
                        input_tokens=60_000_000,
                        output_tokens=90_000_000,
                        estimated_output_bound=_WIRE_QUANTITY_MAX,
                        receipt_aggregate_count=_WIRE_QUANTITY_MAX,
                        receipt_pricing_input_tokens=60_000_000,
                    ),
                    "retry_exhausted",
                )

            reporter._drain_receipt_folds_to_queue(final=True)
            replays = [pending.event for pending in reporter._queue]
            assert len(replays) == 3
            assert len({event.call_id for event in replays}) == 3
            for event in replays:
                assert event.receipt_aggregate_count is not None
                assert 1 <= event.receipt_aggregate_count <= _WIRE_QUANTITY_MAX
                assert event.input_tokens <= _WIRE_QUANTITY_MAX
                assert event.output_tokens <= _WIRE_QUANTITY_MAX
                assert (event.estimated_output_bound or 0) <= _WIRE_QUANTITY_MAX
                assert event.receipt_pricing_input_tokens == 60_000_000
            assert sum(event.receipt_aggregate_count or 0 for event in replays) == 300_000_000
            assert sum(event.input_tokens for event in replays) == 180_000_000
            assert sum(event.output_tokens for event in replays) == 270_000_000
            assert sum(event.estimated_output_bound or 0 for event in replays) == 300_000_000

            refolded = _quiet_sync()
            try:
                for event in replays:
                    refolded._fold_or_count_event_drop(event, "retry_exhausted")
                refolded._drain_receipt_folds_to_queue(final=True)
                second = [pending.event for pending in refolded._queue]
                assert len(second) == 3
                assert sum(event.receipt_aggregate_count or 0 for event in second) == 300_000_000
                assert sum(event.input_tokens for event in second) == 180_000_000
                assert sum(event.output_tokens for event in second) == 270_000_000
                assert {event.receipt_pricing_input_tokens for event in second} == {60_000_000}
            finally:
                refolded._shutdown.set()
                refolded._http.close()
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_split_reserves_one_count_per_chunk_when_totals_are_not_proportional(
        self,
    ) -> None:
        reporter = _quiet_sync()
        try:
            for suffix in ("a", "b", "c"):
                reporter._fold_or_count_event_drop(
                    _event(
                        call_id=call_uuid(f"non-proportional-{suffix}"),
                        input_tokens=_WIRE_QUANTITY_MAX,
                        output_tokens=1,
                        estimated_output_bound=1,
                    ),
                    "retry_exhausted",
                )

            reporter._drain_receipt_folds_to_queue(final=True)
            replays = [pending.event for pending in reporter._queue]
            # Input alone forces three chunks while every other total fits in
            # one. Each chunk must still RESERVE a receipt count for the chunks
            # after it — take the whole count first and the tail chunks would
            # carry 0 and fail the wire model's ge=1 bound.
            assert len(replays) == 3
            assert [event.receipt_aggregate_count for event in replays] == [1, 1, 1]
            assert [event.input_tokens for event in replays] == [_WIRE_QUANTITY_MAX] * 3
            assert [event.output_tokens for event in replays] == [3, 0, 0]
            assert [event.estimated_output_bound for event in replays] == [3, 0, 0]
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    @pytest.mark.parametrize(
        "media_field",
        [
            "image_count",
            "generation_count",
            "video_seconds",
            "audio_seconds",
            "input_characters",
        ],
    )
    def test_media_presence_and_pricing_basis_survive_100m_split_and_refold(
        self, media_field: str
    ) -> None:
        reporter = _quiet_sync()
        quantity = 75_000_000.0 if media_field.endswith("seconds") else 75_000_000
        modality = "video" if media_field == "video_seconds" else "audio"
        quantity_fields = {
            "image_count",
            "generation_count",
            "video_seconds",
            "audio_seconds",
            "input_characters",
        }
        try:
            for suffix in ("a", "b", "c"):
                reporter._fold_or_count_event_drop(
                    _event(
                        call_id=call_uuid(f"{media_field}-100m-{suffix}"),
                        modality=modality,
                        input_tokens=60_000_000,
                        output_tokens=90_000_000,
                        estimated_output_bound=_WIRE_QUANTITY_MAX,
                        receipt_aggregate_count=_WIRE_QUANTITY_MAX,
                        receipt_pricing_input_tokens=150_000,
                        media_usage=MediaUsage(**{media_field: quantity}),
                    ),
                    "retry_exhausted",
                )

            reporter._drain_receipt_folds_to_queue(final=True)
            first = [pending.event for pending in reporter._queue]
            assert len(first) == 3
            assert {event.receipt_pricing_input_tokens for event in first} == {150_000}
            assert (
                sum(getattr(event.media_usage, media_field) or 0 for event in first) == 225_000_000
            )
            for event in first:
                assert event.media_usage is not None
                media_quantity = getattr(event.media_usage, media_field)
                assert media_quantity is not None
                assert media_quantity <= _WIRE_QUANTITY_MAX
                media_payload = event.model_dump(mode="json", exclude_none=True)["media_usage"]
                assert quantity_fields.intersection(media_payload) == {media_field}

            refolded = _quiet_sync()
            try:
                for event in first:
                    refolded._fold_or_count_event_drop(event, "retry_exhausted")
                refolded._drain_receipt_folds_to_queue(final=True)
                second = [pending.event for pending in refolded._queue]
                assert len(second) == 3
                assert {event.receipt_pricing_input_tokens for event in second} == {150_000}
                assert (
                    sum(getattr(event.media_usage, media_field) or 0 for event in second)
                    == 225_000_000
                )
                assert all(
                    event.media_usage is not None
                    and getattr(event.media_usage, media_field) is not None
                    and getattr(event.media_usage, media_field) <= _WIRE_QUANTITY_MAX
                    for event in second
                )
                for event in second:
                    media_payload = event.model_dump(mode="json", exclude_none=True)["media_usage"]
                    assert quantity_fields.intersection(media_payload) == {media_field}
            finally:
                refolded._shutdown.set()
                refolded._http.close()
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_fold_retains_no_content_or_exception_strings(self) -> None:
        reporter = _quiet_sync()
        secret = "TOP-SECRET-PROMPT-CONTENT"
        try:
            reporter._fold_or_count_event_drop(
                _event(
                    deny_reason=secret,
                    tags={"customer-content": secret},
                    failover_error_class=secret,
                    agent_run_name=secret,
                ),
                "retry_exhausted",
            )
            state = repr(dataclasses.asdict(_only_fold(reporter)))
            assert secret not in state
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_concurrent_updates_to_one_key_are_lossless(self) -> None:
        reporter = _quiet_sync()

        def fold_many() -> None:
            for _ in range(100):
                reporter._fold_or_count_event_drop(_event(input_tokens=2), "retry_exhausted")

        threads = [threading.Thread(target=fold_many) for _ in range(8)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)
            assert all(not thread.is_alive() for thread in threads)
            fold = _only_fold(reporter)
            assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (
                800,
                1600,
                5600,
            )
        finally:
            reporter._shutdown.set()
            reporter._http.close()


@pytest.mark.unit
class TestSyncReceiptFolding:
    def test_retry_exhausted_denial_folds_without_ordinary_drop_count(self) -> None:
        reporter = _quiet_sync(max_send_attempts=1)
        reporter._queue.append(_PendingEvent(_event(input_tokens=9, estimated_output_bound=17)))
        try:
            with patch.object(reporter._http, "post", side_effect=httpx.ConnectError("down")):
                reporter._flush_remaining()

            fold = _only_fold(reporter)
            assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (1, 9, 17)
            assert "event.retry_exhausted" not in reporter.dropped_counts
            assert not reporter._queue
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_hundred_denials_fold_then_recover_on_cycle_after_success(self) -> None:
        reporter = _quiet_sync(batch_size=100, max_send_attempts=1)
        for i in range(100):
            reporter._queue.append(
                _PendingEvent(
                    _event(
                        call_id=call_uuid(f"dead-{i}"),
                        input_tokens=1,
                        estimated_output_bound=2,
                    )
                )
            )
        try:
            with patch.object(reporter._http, "post", side_effect=httpx.ConnectError("down")):
                reporter._flush_remaining()
            fold = _only_fold(reporter)
            assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (
                100,
                100,
                200,
            )

            payloads: list[list[dict[str, object]]] = []

            def post(_url: str, **kwargs: object) -> MagicMock:
                payloads.append(kwargs["json"])  # type: ignore[arg-type]
                return _ok_response()

            ordinary_id = call_uuid("recovery-probe")
            reporter.report(
                _event(
                    call_id=ordinary_id,
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
            with patch.object(reporter._http, "post", side_effect=post):
                reporter._flush_remaining()  # establishes recovery; fold stays put
                assert len(payloads) == 1
                assert payloads[0][0]["call_id"] == ordinary_id
                assert len(reporter._receipt_fold_snapshot()) == 1
                reporter._flush_remaining()  # previous success drains the fold

            assert len(payloads) == 2
            [aggregate] = payloads[1]
            assert aggregate["status"] == "budget_denied"
            assert aggregate["deny_source"] == "aggregate_replay"
            assert aggregate["receipt_aggregate_count"] == 100
            assert aggregate["input_tokens"] == 100
            assert aggregate["receipt_pricing_input_tokens"] == 1
            assert aggregate["estimated_output_bound"] == 200
            assert aggregate["agent_run_id"] == "run-fold"
            assert aggregate["model"] == "gpt-5.5"
            assert aggregate["provider"] == "openai"
            original_ids = {ordinary_id, *[call_uuid(f"dead-{i}") for i in range(100)]}
            assert aggregate["call_id"] not in original_ids
            assert datetime.fromisoformat(str(aggregate["timestamp"])).tzinfo is not None
            assert reporter._receipt_fold_snapshot() == {}
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_failed_aggregate_refolds_existing_totals_without_double_count(self) -> None:
        reporter = _quiet_sync(max_send_attempts=1)
        aggregate = _event(
            deny_source="aggregate_replay",
            receipt_aggregate_count=100,
            input_tokens=500,
            estimated_output_bound=700,
        )
        reporter._queue.append(_PendingEvent(aggregate))
        try:
            with patch.object(reporter._http, "post", side_effect=httpx.ConnectError("down")):
                reporter._flush_remaining()
            fold = _only_fold(reporter)
            assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (100, 500, 700)

            sent: list[list[dict[str, object]]] = []

            def post(_url: str, **kwargs: object) -> MagicMock:
                sent.append(kwargs["json"])  # type: ignore[arg-type]
                return _ok_response()

            reporter.report(
                _event(
                    call_id=call_uuid("aggregate-recovery"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
            with patch.object(reporter._http, "post", side_effect=post):
                reporter._flush_remaining()
                reporter._flush_remaining()
            replay = sent[-1][0]
            assert (replay["receipt_aggregate_count"], replay["input_tokens"]) == (100, 500)
            assert replay["estimated_output_bound"] == 700
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_queue_overflow_folds_denied_but_counts_success(self) -> None:
        reporter = _quiet_sync(max_queue_size=1)
        try:
            reporter.report(_event(call_id=call_uuid("evicted-denied")))
            reporter.report(
                _event(
                    call_id=call_uuid("evicting-success"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
            assert _only_fold(reporter).count == 1
            assert "event.overflow" not in reporter.dropped_counts

            reporter.report(_event(call_id=call_uuid("evicts-success")))
            assert reporter.dropped_counts["event.overflow"] == 1
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_terminal_status_and_deadline_sweeps_fold_only_denials(self) -> None:
        reporter = _quiet_sync(batch_size=10)
        denied = _PendingEvent(_event(call_id=call_uuid("terminal-denied")))
        success = _PendingEvent(
            _event(
                call_id=call_uuid("terminal-success"),
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        )
        reporter._queue.extend((denied, success))
        response = MagicMock(spec=httpx.Response)
        response.status_code = 422
        response.raise_for_status = MagicMock(
            spec=httpx.Response.raise_for_status,
            side_effect=httpx.HTTPStatusError(
                "terminal",
                request=MagicMock(spec=httpx.Request),
                response=response,
            ),
        )
        try:
            with patch.object(reporter._http, "post", return_value=response):
                reporter._flush_remaining()
            assert _only_fold(reporter).count == 1
            assert reporter.dropped_counts["event.terminal_status"] == 1

            reporter._queue.extend(
                (
                    _PendingEvent(_event(call_id=call_uuid("deadline-denied"))),
                    _PendingEvent(
                        _event(
                            call_id=call_uuid("deadline-success"),
                            status=CallStatus.SUCCESS,
                            deny_source=None,
                            deny_reason=None,
                            estimated_output_bound=None,
                        )
                    ),
                )
            )
            reporter._drain_event_batches(deadline=0.0, final=True)
            assert _only_fold(reporter).count == 2
            assert reporter.dropped_counts["event.shutdown_deadline"] == 1
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_indexed_ingest_rejection_folds_denied_member_only(self) -> None:
        reporter = _quiet_sync(batch_size=10)
        reporter._queue.extend(
            (
                _PendingEvent(_event(call_id=call_uuid("reject-denied"))),
                _PendingEvent(
                    _event(
                        call_id=call_uuid("accept-success"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                ),
            )
        )
        response = _ok_response()
        response.json.return_value = {
            "ingested": 1,
            "rejected": [
                {
                    "index": 0,
                    "code": "unknown_model",
                    "model": "gpt-5.5",
                    "message": "structural",
                }
            ],
        }
        try:
            with patch.object(reporter._http, "post", return_value=response):
                reporter._flush_remaining()
            assert _only_fold(reporter).count == 1
            assert "event.ingest_rejected" not in reporter.dropped_counts
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_full_legacy_rejection_disposes_aggregate_with_original_cardinality(self) -> None:
        reporter = _quiet_sync(batch_size=1)
        reporter._queue.append(
            _PendingEvent(
                _event(
                    call_id=call_uuid("legacy-full-aggregate"),
                    deny_source="aggregate_replay",
                    receipt_aggregate_count=100,
                    input_tokens=500,
                    estimated_output_bound=700,
                )
            )
        )
        response = _ok_response()
        response.json.return_value = {
            "ingested": 0,
            "rejected": [
                {
                    "code": "unknown_model",
                    "model": "gpt-5.5",
                    "message": "structural",
                }
            ],
        }
        try:
            with patch.object(reporter._http, "post", return_value=response):
                assert reporter._drain_event_batches() is False

            fold = _only_fold(reporter)
            assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (100, 500, 700)
            assert reporter.dropped_counts == {}

            # A rejected batch is not a recovery proof, so the aggregate is
            # not immediately injected into another failing cycle.
            reporter._drain_receipt_folds_to_queue(final=False)
            assert not reporter._queue
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_partial_legacy_rejection_charges_the_heaviest_receipt_weights(self) -> None:
        reporter = _quiet_sync(batch_size=2)
        reporter._queue.extend(
            (
                _PendingEvent(
                    _event(
                        call_id=call_uuid("legacy-partial-aggregate"),
                        deny_source="aggregate_replay",
                        receipt_aggregate_count=100,
                    )
                ),
                _PendingEvent(
                    _event(
                        call_id=call_uuid("legacy-partial-success"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                ),
            )
        )
        response = _ok_response()
        response.json.return_value = {
            "ingested": 1,
            "rejected": [
                {
                    "code": "unknown_model",
                    "model": "gpt-5.5",
                    "message": "structural",
                }
            ],
        }
        try:
            with patch.object(reporter._http, "post", return_value=response):
                reporter._drain_event_batches()
            assert reporter._receipt_fold_snapshot() == {}
            # An index-less body proves ONE rejection but not which event: the
            # heaviest candidate (the 100-receipt aggregate) is charged, never
            # the single raw event that would understate the loss by 99.
            assert reporter.dropped_counts == {"event.ingest_rejected": 100}
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    @pytest.mark.parametrize("invalid_index", [0.9, True])
    def test_non_integer_rejection_index_degrades_to_legacy_count_only(
        self, invalid_index: object
    ) -> None:
        reporter = _quiet_sync(batch_size=2)
        reporter._queue.extend(
            (
                _PendingEvent(_event(call_id=call_uuid("malformed-index-denied"))),
                _PendingEvent(
                    _event(
                        call_id=call_uuid("malformed-index-success"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                ),
            )
        )
        response = _ok_response()
        response.json.return_value = {
            "ingested": 1,
            "rejected": [
                {
                    "index": invalid_index,
                    "code": "unknown_model",
                    "model": "gpt-5.5",
                    "message": "structural",
                }
            ],
        }
        try:
            with patch.object(reporter._http, "post", return_value=response):
                reporter._drain_event_batches()
            # A server-side index bug must not zero the loss the pre-index
            # parser counted: identity is unusable, the count still stands.
            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {"event.ingest_rejected": 1}
            assert not reporter._queue
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_rejected_aggregate_waits_for_a_later_clean_cycle_before_replay(self) -> None:
        reporter = _quiet_sync()
        reporter._fold_or_count_event_drop(
            _event(receipt_aggregate_count=100, input_tokens=500, estimated_output_bound=700),
            "retry_exhausted",
        )
        payloads: list[list[dict[str, object]]] = []
        aggregate_rejections = 0

        def post(_url: str, **kwargs: object) -> MagicMock:
            nonlocal aggregate_rejections
            payload = kwargs["json"]  # type: ignore[assignment]
            payloads.append(payload)  # type: ignore[arg-type]
            response = _ok_response()
            aggregate_index = next(
                (
                    index
                    for index, event in enumerate(payload)
                    if event.get("deny_source") == "aggregate_replay"
                ),
                None,
            )
            if aggregate_index is not None and not aggregate_rejections:
                aggregate_rejections += 1
                response.json.return_value = {
                    "ingested": len(payload) - 1,
                    "rejected": [
                        {
                            "index": aggregate_index,
                            "code": "unknown_model",
                            "model": "gpt-5.5",
                            "message": "structural",
                        }
                    ],
                }
            return response

        try:
            reporter.report(
                _event(
                    call_id=call_uuid("clean-before-rejected-replay"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
            with patch.object(reporter._http, "post", side_effect=post):
                reporter._flush_remaining()  # clean recovery proof
                reporter.report(
                    _event(
                        call_id=call_uuid("accepted-beside-rejected-replay"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                )
                reporter._flush_remaining()  # mixed batch rejects only the replay
                reporter._flush_remaining()  # rejection is not a recovery proof

                assert len(payloads) == 2
                fold = _only_fold(reporter)
                assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (
                    100,
                    500,
                    700,
                )

                reporter.report(
                    _event(
                        call_id=call_uuid("clean-after-rejected-replay"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                )
                reporter._flush_remaining()  # later clean recovery proof
                reporter._flush_remaining()  # replay is eligible once more

            assert len(payloads) == 4
            assert payloads[-1][0]["deny_source"] == "aggregate_replay"
            assert payloads[-1][0]["receipt_aggregate_count"] == 100
            assert reporter._receipt_fold_snapshot() == {}
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_indexed_rejection_and_seal_publish_each_event_once(self) -> None:
        reporter = _quiet_sync(batch_size=10)
        reporter._queue.extend(
            (
                _PendingEvent(_event(call_id=call_uuid("race-denied"))),
                _PendingEvent(
                    _event(
                        call_id=call_uuid("race-success"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                ),
            )
        )
        response = _ok_response()
        response.json.return_value = {
            "ingested": 0,
            "rejected": [
                {
                    "index": 0,
                    "code": "unknown_model",
                    "model": "gpt-5.5",
                    "message": "structural",
                },
                {
                    "index": 1,
                    "code": "unknown_model",
                    "model": "gpt-5.5",
                    "message": "structural",
                },
            ],
        }
        parsed = threading.Event()
        release = threading.Event()
        original_resolve = reporter._resolve_sent_event_batch

        def gated_resolve(*args: object, **kwargs: object):
            if not parsed.is_set():
                parsed.set()
                assert release.wait(timeout=2.0)
            return original_resolve(*args, **kwargs)  # type: ignore[arg-type]

        try:
            with (
                patch.object(reporter._http, "post", return_value=response),
                patch.object(reporter, "_resolve_sent_event_batch", side_effect=gated_resolve),
            ):
                worker = threading.Thread(target=reporter._flush_remaining)
                worker.start()
                assert parsed.wait(timeout=2.0)
                reporter._seal_delivery()
                release.set()
                worker.join(timeout=2.0)
                assert not worker.is_alive()

            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {"event.shutdown_deadline": 2}
        finally:
            release.set()
            reporter._shutdown.set()
            reporter._http.close()

    def test_close_final_flush_resolves_exact_rejection_against_its_own_batch(self) -> None:
        reporter = _quiet_sync(batch_size=1)
        blocked = _event(
            call_id=call_uuid("token-blocked-denied"),
            agent_run_id="token-run-a",
            deny_source="local_enforcement",
            input_tokens=11,
        )
        rejected = _event(
            call_id=call_uuid("token-final-success"),
            status=CallStatus.SUCCESS,
            deny_source=None,
            deny_reason=None,
            estimated_output_bound=None,
        )
        reporter._queue.append(_PendingEvent(blocked))
        blocked_in_post = threading.Event()
        release_blocked = threading.Event()
        seal_ready = threading.Event()
        allow_seal = threading.Event()
        original_seal = reporter._seal_delivery

        def post(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]  # type: ignore[assignment]
            call_id = payload[0]["call_id"]
            if call_id == blocked.call_id:
                blocked_in_post.set()
                assert release_blocked.wait(timeout=5.0)
                return _ok_response()
            assert call_id == rejected.call_id
            response = _ok_response()
            response.json.return_value = {
                "ingested": 0,
                "rejected": [
                    {
                        "index": 0,
                        "code": "unknown_model",
                        "model": rejected.model,
                        "message": "structural",
                    }
                ],
            }
            return response

        def gated_seal(*, emit: bool = True) -> bool:
            seal_ready.set()
            assert allow_seal.wait(timeout=5.0)
            return original_seal(emit=emit)

        background = threading.Thread(target=reporter._flush_remaining)
        closer = threading.Thread(target=lambda: reporter.close(timeout=2.0))
        try:
            with (
                patch.object(reporter._http, "post", side_effect=post),
                patch.object(reporter, "_seal_delivery", side_effect=gated_seal),
            ):
                background.start()
                assert blocked_in_post.wait(timeout=2.0)
                reporter.report(rejected)
                closer.start()
                assert seal_ready.wait(timeout=2.0)

                # Final flush B resolved before blocked batch A. B's indexed
                # rejection must count B, never fold A by consuming a global
                # event-identity FIFO.
                assert reporter.dropped_counts.get("event.ingest_rejected") == 1
                assert reporter._receipt_fold_snapshot() == {}

                allow_seal.set()
                closer.join(timeout=2.0)
                assert not closer.is_alive()
                release_blocked.set()
                background.join(timeout=2.0)
                assert not background.is_alive()

            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {
                "event.ingest_rejected": 1,
                "event.shutdown_deadline": 1,
            }
            assert reporter._receipt_fold_state.previous_cycle_succeeded is False
        finally:
            release_blocked.set()
            allow_seal.set()
            background.join(timeout=2.0)
            closer.join(timeout=2.0)
            reporter._shutdown.set()
            reporter._http.close()

    def test_out_of_order_terminal_exact_and_retry_requeue_keep_batch_identity(self) -> None:
        reporter = _quiet_sync(batch_size=1)
        blocked = _event(
            call_id=call_uuid("token-retry-success"),
            status=CallStatus.SUCCESS,
            deny_source=None,
            deny_reason=None,
            estimated_output_bound=None,
        )
        terminal = _event(
            call_id=call_uuid("token-terminal-denied"),
            agent_run_id="token-run-b",
            deny_source="server",
        )
        exact = _event(
            call_id=call_uuid("token-exact-denied"),
            agent_run_id="token-run-c",
            deny_source="local_velocity",
        )
        reporter._queue.append(_PendingEvent(blocked))
        blocked_in_post = threading.Event()
        release_retry = threading.Event()
        seal_ready = threading.Event()
        allow_seal = threading.Event()
        original_seal = reporter._seal_delivery

        def post(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]  # type: ignore[assignment]
            call_id = payload[0]["call_id"]
            if call_id == blocked.call_id:
                blocked_in_post.set()
                assert release_retry.wait(timeout=5.0)
                raise httpx.ConnectError("retry")
            if call_id == terminal.call_id:
                response = MagicMock(spec=httpx.Response)
                response.status_code = 422
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "terminal",
                    request=MagicMock(spec=httpx.Request),
                    response=response,
                )
                return response
            assert call_id == exact.call_id
            response = _ok_response()
            response.json.return_value = {
                "ingested": 0,
                "rejected": [
                    {
                        "index": 0,
                        "code": "unknown_model",
                        "model": exact.model,
                        "message": "structural",
                    }
                ],
            }
            return response

        def gated_seal(*, emit: bool = True) -> bool:
            seal_ready.set()
            assert allow_seal.wait(timeout=5.0)
            return original_seal(emit=emit)

        background = threading.Thread(target=reporter._flush_remaining)
        closer = threading.Thread(target=lambda: reporter.close(timeout=2.0))
        try:
            with (
                patch.object(reporter._http, "post", side_effect=post),
                patch.object(reporter, "_seal_delivery", side_effect=gated_seal),
            ):
                background.start()
                assert blocked_in_post.wait(timeout=2.0)
                reporter.report(terminal)
                reporter.report(exact)
                closer.start()
                assert seal_ready.wait(timeout=2.0)

                # close's two final-flush passes resolved B terminally and C
                # by exact rejection while A remained blocked.
                assert reporter._receipt_fold_snapshot() == {}
                assert reporter.dropped_counts == {
                    "event.terminal_status": 1,
                    "event.ingest_rejected": 1,
                }

                # Let A's retry disposition requeue its own prefix before the
                # close seal claims the remaining queue.
                release_retry.set()
                background.join(timeout=2.0)
                assert not background.is_alive()
                assert [pending.event.call_id for pending in reporter._queue] == [blocked.call_id]
                allow_seal.set()
                closer.join(timeout=2.0)
                assert not closer.is_alive()

            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {
                "event.terminal_status": 1,
                "event.ingest_rejected": 1,
                "event.shutdown_deadline": 1,
            }
            assert reporter._receipt_fold_state.previous_cycle_succeeded is False
        finally:
            release_retry.set()
            allow_seal.set()
            background.join(timeout=2.0)
            closer.join(timeout=2.0)
            reporter._shutdown.set()
            reporter._http.close()

    @pytest.mark.parametrize(
        ("final_outcome", "final_drop_reason"),
        [
            (_SendOutcome.SENT, None),
            (_SendOutcome.RETRY, "settlement_confirm.retry_exhausted"),
            (_SendOutcome.HELD, "settlement_confirm.exit_breaker_open"),
            (_SendOutcome.DROPPED, "settlement_confirm.terminal_status"),
        ],
    )
    @pytest.mark.parametrize("blocked_denied", [True, False])
    def test_close_resolves_out_of_order_settlements_by_exact_pair(
        self,
        final_outcome: _SendOutcome,
        final_drop_reason: str | None,
        *,
        blocked_denied: bool,
    ) -> None:
        reporter = _quiet_sync(batch_size=1)
        blocked_values: dict[str, object] = {
            "call_id": call_uuid(f"settlement-blocked-{blocked_denied}"),
            "agent_run_id": "settlement-run-a",
            "deny_source": "local_enforcement",
        }
        final_values: dict[str, object] = {
            "call_id": call_uuid(f"settlement-final-{blocked_denied}-{final_outcome.value}"),
            "agent_run_id": "settlement-run-b",
            "deny_source": "server",
        }
        success_values: dict[str, object] = {
            "status": CallStatus.SUCCESS,
            "deny_source": None,
            "deny_reason": None,
            "estimated_output_bound": None,
        }
        (final_values if blocked_denied else blocked_values).update(success_values)
        blocked = _event(**blocked_values)
        final = _event(**final_values)
        blocked_confirm = _confirm(
            reservation_id="res_settlement_a",
            call_id=blocked.call_id,
        )
        final_confirm = _confirm(
            reservation_id="res_settlement_b",
            call_id=final.call_id,
        )
        reporter.report_settlement(blocked_confirm, blocked)
        blocked_in_post = threading.Event()
        release_blocked = threading.Event()
        seal_ready = threading.Event()
        allow_seal = threading.Event()
        ingested: list[str] = []
        original_seal = reporter._seal_delivery

        def send_confirm(
            request: BudgetConfirmRequest, *, timeout: float | None = None
        ) -> _SendOutcome:
            del timeout
            if request.call_id == blocked.call_id:
                blocked_in_post.set()
                assert release_blocked.wait(timeout=5.0)
                return _SendOutcome.SENT
            assert request.call_id == final.call_id
            return final_outcome

        def post(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]  # type: ignore[assignment]
            assert payload[0]["call_id"] == final.call_id
            ingested.append(payload[0]["call_id"])
            return _ok_response()

        def gated_seal(*, emit: bool = True) -> bool:
            seal_ready.set()
            assert allow_seal.wait(timeout=5.0)
            return original_seal(emit=emit)

        background = threading.Thread(target=reporter._flush_remaining)
        closer = threading.Thread(target=lambda: reporter.close(timeout=2.0))
        try:
            with (
                patch.object(reporter, "_send_confirm", side_effect=send_confirm),
                patch.object(reporter._http, "post", side_effect=post),
                patch.object(reporter, "_seal_delivery", side_effect=gated_seal),
            ):
                background.start()
                assert blocked_in_post.wait(timeout=2.0)
                reporter.report_settlement(final_confirm, final)
                closer.start()
                assert seal_ready.wait(timeout=2.0)

                # Final drain B resolved and handed off B's exact event while
                # background drain A still owns the first settlement pair.
                assert ingested == [final.call_id]
                allow_seal.set()
                closer.join(timeout=2.0)
                assert not closer.is_alive()
                release_blocked.set()
                background.join(timeout=2.0)
                assert not background.is_alive()

            expected_drops = {"settlement_confirm.shutdown_deadline": 1}
            if final_drop_reason is not None:
                expected_drops[final_drop_reason] = 1
            assert reporter._receipt_fold_snapshot() == {}
            expected_drops["event.shutdown_deadline"] = 1
            assert reporter.dropped_counts == expected_drops
            assert ingested == [final.call_id]
        finally:
            release_blocked.set()
            allow_seal.set()
            background.join(timeout=2.0)
            closer.join(timeout=2.0)
            reporter._shutdown.set()
            reporter._http.close()

    @pytest.mark.parametrize("park_outcome", [None, _SendOutcome.RETRY, _SendOutcome.HELD])
    @pytest.mark.parametrize("blocked_denied", [True, False])
    def test_out_of_order_settlement_parking_keeps_exact_pair(
        self,
        park_outcome: _SendOutcome | None,
        *,
        blocked_denied: bool,
    ) -> None:
        reporter = _quiet_sync(batch_size=1)
        blocked_values: dict[str, object] = {
            "call_id": call_uuid(f"settlement-park-blocked-{blocked_denied}"),
            "agent_run_id": "settlement-park-run-a",
            "deny_source": "local_enforcement",
        }
        parked_values: dict[str, object] = {
            "call_id": call_uuid(f"settlement-parked-{blocked_denied}-{park_outcome}"),
            "agent_run_id": "settlement-park-run-b",
            "deny_source": "server",
        }
        success_values: dict[str, object] = {
            "status": CallStatus.SUCCESS,
            "deny_source": None,
            "deny_reason": None,
            "estimated_output_bound": None,
        }
        (parked_values if blocked_denied else blocked_values).update(success_values)
        blocked = _event(**blocked_values)
        parked = _event(**parked_values)
        reporter.report_settlement(
            _confirm(reservation_id="res_park_a", call_id=blocked.call_id), blocked
        )
        blocked_in_post = threading.Event()
        release_blocked = threading.Event()

        def send_confirm(
            request: BudgetConfirmRequest, *, timeout: float | None = None
        ) -> _SendOutcome:
            del timeout
            if request.call_id == blocked.call_id:
                blocked_in_post.set()
                assert release_blocked.wait(timeout=5.0)
                return _SendOutcome.SENT
            assert request.call_id == parked.call_id
            assert park_outcome is not None
            return park_outcome

        background = threading.Thread(target=reporter._flush_remaining)
        try:
            with patch.object(reporter, "_send_confirm", side_effect=send_confirm):
                background.start()
                assert blocked_in_post.wait(timeout=2.0)
                reporter.report_settlement(
                    _confirm(reservation_id="res_park_b", call_id=parked.call_id), parked
                )
                if park_outcome is None:
                    reporter._settlement_queue[-1].confirm.next_attempt_at = float("inf")
                reporter._drain_settlements()
                assert [pending.event.call_id for pending in reporter._settlement_queue] == [
                    parked.call_id
                ]

                # Seal must claim queued B and token-owned A once each. A's
                # late response then owns nothing and publishes nothing.
                reporter._seal_delivery()
                release_blocked.set()
                background.join(timeout=2.0)
                assert not background.is_alive()

            expected_drops = {"settlement_confirm.shutdown_deadline": 2}
            assert reporter._receipt_fold_snapshot() == {}
            expected_drops["event.shutdown_deadline"] = 2
            assert reporter.dropped_counts == expected_drops
        finally:
            release_blocked.set()
            background.join(timeout=2.0)
            reporter._shutdown.set()
            reporter._http.close()

    @pytest.mark.parametrize(
        ("confirm_outcome", "confirm_drop_reason"),
        [
            (_SendOutcome.SENT, None),
            (_SendOutcome.RETRY, "settlement_confirm.retry_exhausted"),
            (_SendOutcome.HELD, "settlement_confirm.exit_breaker_open"),
            (_SendOutcome.DROPPED, "settlement_confirm.terminal_status"),
        ],
    )
    @pytest.mark.parametrize("denied", [True, False])
    def test_close_never_observes_a_gap_between_settlement_token_and_event_disposition(
        self,
        confirm_outcome: _SendOutcome,
        confirm_drop_reason: str | None,
        *,
        denied: bool,
    ) -> None:
        reporter = _quiet_sync(batch_size=1, max_send_attempts=1)
        values: dict[str, object] = {
            "call_id": call_uuid(f"atomic-settlement-{denied}-{confirm_outcome.value}"),
            "agent_run_id": "atomic-settlement-run",
            "deny_source": "server",
        }
        if not denied:
            values.update(
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        event = _event(**values)
        reporter.report_settlement(_confirm(call_id=event.call_id), event)
        worker_name = f"settlement-transition-{denied}-{confirm_outcome.value}"
        gate = _GateAfterNthOwnershipRelease(worker_name, 2)
        reporter._ownership_lock = gate  # type: ignore[assignment]
        worker = threading.Thread(
            target=lambda: reporter._drain_settlements(final=True),
            name=worker_name,
        )
        closer = threading.Thread(target=lambda: reporter.close(timeout=1.0))
        try:
            with (
                patch.object(reporter, "_send_confirm", return_value=confirm_outcome),
                patch.object(reporter._http, "post", side_effect=httpx.ConnectError("down")),
            ):
                worker.start()
                assert gate.released.wait(timeout=2.0)
                closer.start()
                closer.join(timeout=2.0)
                assert not closer.is_alive()

                expected_drops: dict[str, int] = {}
                if confirm_drop_reason is not None:
                    expected_drops[confirm_drop_reason] = 1
                assert reporter._receipt_fold_snapshot() == {}
                expected_drops["event.retry_exhausted"] = 1
                assert reporter.dropped_counts == expected_drops

                # The late outcome handler no longer owns a token and must not
                # add a second successor or disposition after close returned.
                before_folds = reporter._receipt_fold_snapshot()
                gate.allow_worker.set()
                worker.join(timeout=2.0)
                assert not worker.is_alive()
                assert reporter._receipt_fold_snapshot() == before_folds
                assert reporter.dropped_counts == expected_drops
        finally:
            gate.allow_worker.set()
            worker.join(timeout=2.0)
            closer.join(timeout=2.0)
            reporter._shutdown.set()
            reporter._http.close()

    @pytest.mark.parametrize("denied", [True, False])
    def test_close_never_observes_a_gap_during_settlement_deadline_disposition(
        self, *, denied: bool
    ) -> None:
        reporter = _quiet_sync(batch_size=1)
        values: dict[str, object] = {
            "call_id": call_uuid(f"atomic-settlement-deadline-{denied}"),
            "agent_run_id": "atomic-deadline-run",
            "deny_source": "server",
        }
        if not denied:
            values.update(
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        event = _event(**values)
        reporter.report_settlement(_confirm(call_id=event.call_id), event)
        worker_name = f"settlement-deadline-{denied}"
        gate = _GateAfterNthOwnershipRelease(worker_name, 1)
        reporter._ownership_lock = gate  # type: ignore[assignment]
        worker = threading.Thread(
            target=lambda: reporter._drain_settlements(deadline=0.0, final=True),
            name=worker_name,
        )
        closer = threading.Thread(target=lambda: reporter.close(timeout=1.0))
        try:
            worker.start()
            assert gate.released.wait(timeout=2.0)
            closer.start()
            closer.join(timeout=2.0)
            assert not closer.is_alive()

            expected_drops = {"settlement_confirm.shutdown_deadline": 1}
            assert reporter._receipt_fold_snapshot() == {}
            expected_drops["event.retry_exhausted" if denied else "event.shutdown_deadline"] = 1
            assert reporter.dropped_counts == expected_drops

            before_folds = reporter._receipt_fold_snapshot()
            gate.allow_worker.set()
            worker.join(timeout=2.0)
            assert not worker.is_alive()
            assert reporter._receipt_fold_snapshot() == before_folds
            assert reporter.dropped_counts == expected_drops
        finally:
            gate.allow_worker.set()
            worker.join(timeout=2.0)
            closer.join(timeout=2.0)
            reporter._shutdown.set()
            reporter._http.close()

    def test_sealed_and_settlement_event_loss_paths_fold_denials(self) -> None:
        reporter = _quiet_sync(max_queue_size=1)
        try:
            reporter._move_event_to_queue(_event(call_id=call_uuid("settlement-denied")))
            reporter._move_event_to_queue(
                _event(
                    call_id=call_uuid("settlement-success"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
            assert _only_fold(reporter).count == 1

            reporter._seal_delivery()
            reporter.report(_event(call_id=call_uuid("sealed-denied")))
            reporter.report_settlement(_confirm(), _event(call_id=call_uuid("sealed-pair")))
            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {
                "event.shutdown_deadline": 2,
                "event.closed_enqueue": 2,
                "settlement_confirm.closed_enqueue": 1,
            }
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_close_replays_one_fold_and_failed_close_is_bounded(self) -> None:
        recovered = _quiet_sync(shutdown_deadline=0.2, max_send_attempts=1)
        recovered._fold_or_count_event_drop(_event(input_tokens=11), "retry_exhausted")
        sent: list[list[dict[str, object]]] = []

        def post(_url: str, **kwargs: object) -> MagicMock:
            sent.append(kwargs["json"])  # type: ignore[arg-type]
            return _ok_response()

        with patch.object(recovered._http, "post", side_effect=post):
            recovered.close()
        assert len(sent) == 1
        assert sent[0][0]["deny_source"] == "aggregate_replay"
        assert recovered._receipt_fold_snapshot() == {}

        failed = _quiet_sync(shutdown_deadline=0.05, max_send_attempts=1)
        failed._fold_or_count_event_drop(_event(input_tokens=13), "retry_exhausted")
        with patch.object(failed._http, "post", side_effect=httpx.ConnectError("down")):
            failed.close()
        assert failed._receipt_fold_snapshot() == {}
        assert failed.dropped_counts["event.retry_exhausted"] == 1

        # Final ownership is terminal: a second close cannot recreate an
        # aggregate that no delivery owner remains alive to send.
        before = failed.dropped_counts
        failed.close()
        assert failed._receipt_fold_snapshot() == {}
        assert failed.dropped_counts == before

    def test_final_close_bypasses_normal_capacity_for_full_fold_snapshot(self) -> None:
        reporter = _quiet_sync(max_queue_size=1, batch_size=50)
        reporter._queue.append(
            _PendingEvent(
                _event(
                    call_id=call_uuid("preexisting-success"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
        )
        for index in range(256):
            reporter._fold_or_count_event_drop(
                _event(agent_run_id=f"final-run-{index}"), "retry_exhausted"
            )
        sent: list[dict[str, object]] = []

        def post(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]  # type: ignore[assignment]
            sent.extend(payload)  # type: ignore[arg-type]
            response = _ok_response()
            response.json.return_value = {"ingested": len(payload), "rejected": []}
            return response

        with patch.object(reporter._http, "post", side_effect=post):
            reporter.close()

        aggregates = [event for event in sent if event.get("deny_source") == "aggregate_replay"]
        assert len(sent) == 257
        assert len(aggregates) == 256
        assert len({event["call_id"] for event in aggregates}) == 256
        assert reporter._receipt_fold_snapshot() == {}
        assert reporter.dropped_counts == {}

    def test_final_deadline_seals_and_counts_every_fold_without_retention(self) -> None:
        reporter = _quiet_sync(max_queue_size=1, batch_size=50)
        reporter._queue.append(
            _PendingEvent(
                _event(
                    call_id=call_uuid("deadline-preexisting-success"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
        )
        for index in range(256):
            reporter._fold_or_count_event_drop(
                _event(
                    agent_run_id=f"deadline-run-{index}",
                    receipt_aggregate_count=4 if index == 0 else 1,
                ),
                "retry_exhausted",
            )

        reporter.close(timeout=0.0)

        assert reporter._receipt_fold_snapshot() == {}
        assert reporter.dropped_counts["event.shutdown_deadline"] == 260

    def test_direct_seal_splits_200m_residual_fold_before_terminal_disposition(self) -> None:
        reporter = _quiet_sync()
        for suffix in ("a", "b"):
            reporter._fold_or_count_event_drop(
                _event(
                    call_id=call_uuid(f"direct-seal-200m-{suffix}"),
                    input_tokens=_WIRE_QUANTITY_MAX,
                    estimated_output_bound=_WIRE_QUANTITY_MAX,
                    receipt_aggregate_count=_WIRE_QUANTITY_MAX,
                    receipt_pricing_input_tokens=3,
                ),
                "retry_exhausted",
            )

        try:
            assert reporter._seal_delivery() is True

            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {"event.shutdown_deadline": 2 * _WIRE_QUANTITY_MAX}
            assert reporter._delivery_is_completed() is True
            assert not reporter._queue
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_expired_close_waiter_splits_200m_residual_before_completion(self) -> None:
        reporter = _quiet_sync()
        for suffix in ("a", "b"):
            reporter._fold_or_count_event_drop(
                _event(
                    call_id=call_uuid(f"expired-waiter-200m-{suffix}"),
                    input_tokens=_WIRE_QUANTITY_MAX,
                    estimated_output_bound=_WIRE_QUANTITY_MAX,
                    receipt_aggregate_count=_WIRE_QUANTITY_MAX,
                    receipt_pricing_input_tokens=3,
                ),
                "retry_exhausted",
            )
        existing_owner = threading.Thread(name="paused-close-owner")
        reporter._close_finalization_owner = existing_owner

        try:
            reporter._finish_close(0.0)

            assert reporter._close_finalization_owner is existing_owner
            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {"event.shutdown_deadline": 2 * _WIRE_QUANTITY_MAX}
            assert reporter._delivery_is_completed() is True
            assert not reporter._queue
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    @pytest.mark.parametrize("denied_first,aggregate_count", [(True, 7), (False, 1)])
    def test_close_deadline_disposes_queued_events_before_releasing_ownership(
        self, denied_first: bool, aggregate_count: int
    ) -> None:
        reporter = _quiet_sync(batch_size=2)
        denied = _PendingEvent(
            _event(
                call_id=call_uuid(f"deadline-denied-{denied_first}"),
                deny_source="aggregate_replay" if aggregate_count > 1 else "server",
                receipt_aggregate_count=aggregate_count,
            )
        )
        success = _PendingEvent(
            _event(
                call_id=call_uuid(f"deadline-success-{denied_first}"),
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        )
        reporter._queue.extend([denied, success] if denied_first else [success, denied])

        disposition_entered = threading.Event()
        disposition_finished = threading.Event()
        release_unsafe_disposition = threading.Event()
        original_dispose = reporter._dispose_pending_events

        def gated_dispose(pending: list[_PendingEvent], reason: str, *, emit: bool = True) -> None:
            if (
                threading.current_thread().name == "solwyn-final-flush"
                and reason == "shutdown_deadline"
            ):
                # Before the fix the final worker has already removed every
                # identity from the queue, but has not registered an owner.
                # Pause only for that unsafe outside-lock shape. The fixed
                # atomic transition invokes the mutation while it still owns
                # the reporter lock and therefore never enters this gap.
                owns_no_lock = reporter._ownership_lock.acquire(blocking=False)
                if owns_no_lock:
                    reporter._ownership_lock.release()
                disposition_entered.set()
                if owns_no_lock:
                    assert release_unsafe_disposition.wait(timeout=5.0)
            original_dispose(pending, reason, emit=emit)
            disposition_finished.set()

        close_finished = threading.Event()

        def close() -> None:
            reporter.close(timeout=0.1)
            close_finished.set()

        closer = threading.Thread(target=close, daemon=True)
        try:
            with (
                patch.object(reporter, "_dispose_pending_events", side_effect=gated_dispose),
                patch.object(
                    reporter,
                    "_deadline_expired",
                    side_effect=lambda _deadline: (
                        threading.current_thread().name == "solwyn-final-flush"
                    ),
                ),
                patch.object(reporter, "_start_breaker_cycle", return_value=None),
            ):
                closer.start()
                assert disposition_entered.wait(timeout=2.0)
                closer.join(timeout=2.0)
                assert close_finished.is_set()

                expected = aggregate_count + 1
                assert reporter._receipt_fold_snapshot() == {}
                assert reporter.dropped_counts == {"event.shutdown_deadline": expected}
                before_late_worker = reporter.dropped_counts

                release_unsafe_disposition.set()
                assert disposition_finished.wait(timeout=2.0)
                assert reporter.dropped_counts == before_late_worker
        finally:
            release_unsafe_disposition.set()
            closer.join(timeout=2.0)
            reporter._shutdown.set()
            reporter._http.close()

    def test_final_take_and_queue_transfer_are_atomic_against_seal(self) -> None:
        reporter = _quiet_sync()
        reporter._fold_or_count_event_drop(_event(receipt_aggregate_count=7), "retry_exhausted")
        building_replay = threading.Event()
        release_replay = threading.Event()
        seal_finished = threading.Event()
        original_replay = reporter._replay_event

        def gated_replay(*args: object, **kwargs: object) -> MetadataEvent:
            building_replay.set()
            assert release_replay.wait(timeout=2.0)
            return original_replay(*args, **kwargs)  # type: ignore[arg-type]

        def seal() -> None:
            reporter._seal_delivery()
            seal_finished.set()

        drainer = threading.Thread(
            target=lambda: reporter._drain_receipt_folds_to_queue(final=True)
        )
        sealer = threading.Thread(target=seal)
        try:
            with patch.object(reporter, "_replay_event", side_effect=gated_replay):
                drainer.start()
                assert building_replay.wait(timeout=2.0)
                sealer.start()
                sealer.join(timeout=0.05)
                assert not seal_finished.is_set()
                release_replay.set()
                drainer.join(timeout=2.0)
                sealer.join(timeout=2.0)

            assert not drainer.is_alive()
            assert not sealer.is_alive()
            assert reporter._receipt_fold_snapshot() == {}
            assert reporter.dropped_counts == {"event.shutdown_deadline": 7}
        finally:
            release_replay.set()
            reporter._shutdown.set()
            reporter._http.close()

    def test_exact_rejection_publishes_all_members_when_drop_logger_raises(self) -> None:
        reporter = _quiet_sync(batch_size=2)
        for index in range(2):
            reporter._queue.append(
                _PendingEvent(
                    _event(
                        call_id=call_uuid(f"raise-handler-{index}"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                )
            )
        response = _ok_response()
        response.json.return_value = {
            "ingested": 0,
            "rejected": [
                {"index": 0, "code": "x", "model": "gpt-5.5", "message": "m"},
                {"index": 1, "code": "x", "model": "gpt-5.5", "message": "m"},
            ],
        }
        try:
            with (
                patch.object(reporter._http, "post", return_value=response),
                patch("solwyn.reporter.logger.warning", side_effect=RuntimeError("handler")),
            ):
                reporter._drain_event_batches()

            assert reporter.dropped_counts == {"event.ingest_rejected": 2}
            assert reporter._event_batches_in_hand == {}
        finally:
            reporter._shutdown.set()
            reporter._http.close()

    def test_drop_logger_can_reenter_report_without_ownership_deadlock(self) -> None:
        reporter = _quiet_sync(batch_size=1)
        reporter._queue.append(
            _PendingEvent(
                _event(
                    call_id=call_uuid("reentrant-rejected"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
        )
        reentrant = _event(
            call_id=call_uuid("reentrant-enqueue"),
            status=CallStatus.SUCCESS,
            deny_source=None,
            deny_reason=None,
            estimated_output_bound=None,
        )
        response = _ok_response()
        response.json.return_value = {
            "ingested": 0,
            "rejected": [{"index": 0, "code": "x", "model": "gpt-5.5", "message": "m"}],
        }
        reentered = threading.Event()

        def warning(message: str, *_args: object, **_kwargs: object) -> None:
            if message.startswith("reporter.spend_events_dropped") and not reentered.is_set():
                reporter.report(reentrant)
                reentered.set()

        with (
            patch.object(reporter._http, "post", side_effect=[response, _ok_response()]),
            patch("solwyn.reporter.logger.warning", side_effect=warning),
        ):
            worker = threading.Thread(target=reporter._drain_event_batches, daemon=True)
            worker.start()
            worker.join(timeout=1.0)

        assert not worker.is_alive()
        assert reentered.is_set()
        assert not reporter._queue
        assert reporter.dropped_counts == {"event.ingest_rejected": 1}
        reporter._shutdown.set()
        reporter._http.close()


@pytest.mark.unit
class TestAsyncReceiptFolding:
    @pytest.mark.asyncio
    async def test_retry_exhaustion_and_recovery_mirror_sync(self) -> None:
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, batch_size=100, max_send_attempts=1)
        for i in range(100):
            reporter._queue.append(
                _PendingEvent(
                    _event(
                        call_id=call_uuid(f"async-dead-{i}"),
                        input_tokens=1,
                        estimated_output_bound=3,
                    )
                )
            )

        async def down(_url: str, **_kwargs: object) -> MagicMock:
            raise httpx.ConnectError("down")

        with patch.object(reporter._http, "post", new=down):
            await reporter._flush_remaining()
        assert (_only_fold(reporter).count, _only_fold(reporter).estimated_output_bound) == (
            100,
            300,
        )
        assert reporter.dropped_counts == {}

        sent: list[list[dict[str, object]]] = []

        async def up(_url: str, **kwargs: object) -> MagicMock:
            sent.append(kwargs["json"])  # type: ignore[arg-type]
            return _ok_response()

        reporter.report(
            _event(
                call_id=call_uuid("async-recovery"),
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        )
        with patch.object(reporter._http, "post", new=up):
            await reporter._flush_remaining()
            await reporter._flush_remaining()
        assert len(sent) == 2
        assert sent[-1][0]["receipt_aggregate_count"] == 100
        assert sent[-1][0]["input_tokens"] == 100
        assert reporter._receipt_fold_snapshot() == {}

        reporter._closed = True
        reporter._close_completed = True
        if reporter._shutdown_event is not None:
            reporter._shutdown_event.set()
        if reporter._flush_task is not None:
            await reporter._flush_task
        if reporter._finalizer is not None:
            reporter._finalizer.detach()
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_rejected_aggregate_requires_later_clean_cycle_before_async_replay(self) -> None:
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
        reporter._fold_or_count_event_drop(
            _event(receipt_aggregate_count=100, input_tokens=500, estimated_output_bound=700),
            "retry_exhausted",
        )
        payloads: list[list[dict[str, object]]] = []
        aggregate_rejections = 0

        async def post(_url: str, **kwargs: object) -> MagicMock:
            nonlocal aggregate_rejections
            payload = kwargs["json"]  # type: ignore[assignment]
            payloads.append(payload)  # type: ignore[arg-type]
            response = _ok_response()
            aggregate_index = next(
                (
                    index
                    for index, event in enumerate(payload)
                    if event.get("deny_source") == "aggregate_replay"
                ),
                None,
            )
            if aggregate_index is not None and not aggregate_rejections:
                aggregate_rejections += 1
                response.json.return_value = {
                    "ingested": len(payload) - 1,
                    "rejected": [
                        {
                            "index": aggregate_index,
                            "code": "unknown_model",
                            "model": "gpt-5.5",
                            "message": "structural",
                        }
                    ],
                }
            return response

        reporter.report(
            _event(
                call_id=call_uuid("async-clean-before-rejected-replay"),
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        )
        with patch.object(reporter._http, "post", new=post):
            await reporter._flush_remaining()
            reporter.report(
                _event(
                    call_id=call_uuid("async-accepted-beside-rejected-replay"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
            await reporter._flush_remaining()
            await reporter._flush_remaining()

            assert len(payloads) == 2
            fold = _only_fold(reporter)
            assert (fold.count, fold.input_tokens, fold.estimated_output_bound) == (100, 500, 700)

            reporter.report(
                _event(
                    call_id=call_uuid("async-clean-after-rejected-replay"),
                    status=CallStatus.SUCCESS,
                    deny_source=None,
                    deny_reason=None,
                    estimated_output_bound=None,
                )
            )
            await reporter._flush_remaining()
            await reporter._flush_remaining()

        assert len(payloads) == 4
        assert payloads[-1][0]["receipt_aggregate_count"] == 100
        assert reporter._receipt_fold_snapshot() == {}
        reporter._closed = True
        reporter._close_completed = True
        if reporter._finalizer is not None:
            reporter._finalizer.detach()
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_overflow_sealed_and_settlement_refusals_mirror_sync(self) -> None:
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, max_queue_size=1)
        reporter.report(_event(call_id=call_uuid("async-denied")))
        reporter.report(
            _event(
                call_id=call_uuid("async-success"),
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        )
        reporter._seal_delivery()
        reporter.report(_event(call_id=call_uuid("async-sealed")))
        reporter.report_settlement(_confirm(), _event(call_id=call_uuid("async-pair")))

        assert reporter._receipt_fold_snapshot() == {}
        assert reporter.dropped_counts == {
            "event.shutdown_deadline": 2,
            "event.closed_enqueue": 2,
            "settlement_confirm.closed_enqueue": 1,
        }

        reporter._closed = True
        reporter._close_completed = True
        if reporter._shutdown_event is not None:
            reporter._shutdown_event.set()
        if reporter._flush_task is not None:
            await reporter._flush_task
        if reporter._finalizer is not None:
            reporter._finalizer.detach()
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_settlement_handoff_overflow_and_aggregate_refold_mirror_sync(self) -> None:
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, max_queue_size=1, max_send_attempts=1)
        reporter._move_event_to_queue(_event(call_id=call_uuid("async-settlement-denied")))
        reporter._move_event_to_queue(
            _event(
                call_id=call_uuid("async-settlement-success"),
                status=CallStatus.SUCCESS,
                deny_source=None,
                deny_reason=None,
                estimated_output_bound=None,
            )
        )
        assert _only_fold(reporter).count == 1

        reporter._queue.clear()
        reporter._queue.append(
            _PendingEvent(
                _event(
                    call_id=call_uuid("async-aggregate-dies"),
                    deny_source="aggregate_replay",
                    receipt_aggregate_count=99,
                    input_tokens=501,
                    estimated_output_bound=701,
                )
            )
        )

        async def down(_url: str, **_kwargs: object) -> MagicMock:
            raise httpx.ConnectError("down")

        with patch.object(reporter._http, "post", new=down):
            await reporter._flush_remaining()
        totals = reporter._receipt_fold_snapshot()
        replay = next(
            fold for key, fold in totals.items() if key[0:2] == ("run-fold", "aggregate_replay")
        )
        assert (replay.count, replay.input_tokens, replay.estimated_output_bound) == (
            99,
            501,
            701,
        )

        reporter._closed = True
        reporter._close_completed = True
        if reporter._finalizer is not None:
            reporter._finalizer.detach()
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_close_replays_fold_once_and_failed_close_counts_terminal_loss(self) -> None:
        recovered = AsyncMetadataReporter(
            _URL, VALID_API_KEY, shutdown_deadline=0.2, max_send_attempts=1
        )
        recovered._fold_or_count_event_drop(_event(input_tokens=17), "retry_exhausted")
        sent: list[list[dict[str, object]]] = []

        async def up(_url: str, **kwargs: object) -> MagicMock:
            sent.append(kwargs["json"])  # type: ignore[arg-type]
            return _ok_response()

        with patch.object(recovered._http, "post", new=up):
            await recovered.close()
        assert len(sent) == 1
        assert sent[0][0]["receipt_aggregate_count"] == 1
        assert recovered._receipt_fold_snapshot() == {}

        failed = AsyncMetadataReporter(
            _URL, VALID_API_KEY, shutdown_deadline=0.05, max_send_attempts=1
        )
        failed._fold_or_count_event_drop(_event(input_tokens=19), "retry_exhausted")

        async def down(_url: str, **_kwargs: object) -> MagicMock:
            raise httpx.ConnectError("down")

        with patch.object(failed._http, "post", new=down):
            await failed.close()
        assert failed._receipt_fold_snapshot() == {}
        assert failed.dropped_counts["event.retry_exhausted"] == 1

        before = failed.dropped_counts
        await failed.close()
        assert failed._receipt_fold_snapshot() == {}
        assert failed.dropped_counts == before

    @pytest.mark.asyncio
    async def test_final_close_bypasses_capacity_and_terminally_counts_failure(self) -> None:
        healthy = AsyncMetadataReporter(
            _URL, VALID_API_KEY, max_queue_size=1, batch_size=50, max_send_attempts=1
        )
        for index in range(3):
            healthy._fold_or_count_event_drop(
                _event(agent_run_id=f"async-final-{index}"), "retry_exhausted"
            )
        sent: list[dict[str, object]] = []

        async def up(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]  # type: ignore[assignment]
            sent.extend(payload)  # type: ignore[arg-type]
            response = _ok_response()
            response.json.return_value = {"ingested": len(payload), "rejected": []}
            return response

        with patch.object(healthy._http, "post", new=up):
            await healthy.close()
        assert len(sent) == 3
        assert healthy._receipt_fold_snapshot() == {}
        assert healthy.dropped_counts == {}

        failed = AsyncMetadataReporter(
            _URL, VALID_API_KEY, max_queue_size=1, batch_size=50, max_send_attempts=1
        )
        for index, count in enumerate((2, 3, 5)):
            failed._fold_or_count_event_drop(
                _event(agent_run_id=f"async-failed-{index}", receipt_aggregate_count=count),
                "retry_exhausted",
            )

        async def down(_url: str, **_kwargs: object) -> MagicMock:
            raise httpx.ConnectError("down")

        with patch.object(failed._http, "post", new=down):
            await failed.close()
        assert failed._receipt_fold_snapshot() == {}
        assert failed.dropped_counts["event.retry_exhausted"] == 10

    @pytest.mark.asyncio
    async def test_deadline_sweep_and_blocking_exit_fold_denials(self) -> None:
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
        reporter._queue.extend(
            (
                _PendingEvent(_event(call_id=call_uuid("async-deadline-denied"))),
                _PendingEvent(
                    _event(
                        call_id=call_uuid("async-deadline-success"),
                        status=CallStatus.SUCCESS,
                        deny_source=None,
                        deny_reason=None,
                        estimated_output_bound=None,
                    )
                ),
            )
        )
        await reporter._drain_event_batches(deadline=0.0, final=True)
        assert _only_fold(reporter).count == 1
        assert reporter.dropped_counts["event.shutdown_deadline"] == 1

        # The loop-less atexit/GC path takes terminal fold ownership. Failed
        # replays are counted, never retained without a future delivery owner.
        reporter._queue.append(_PendingEvent(_event(call_id=call_uuid("async-exit-denied"))))
        sync_client = MagicMock(spec=httpx.Client)
        sync_client.post.side_effect = httpx.ConnectError("down")
        with patch("solwyn._lifecycle.httpx.Client", return_value=sync_client):
            blocking_exit_flush(reporter)
        assert reporter._receipt_fold_snapshot() == {}
        assert reporter.dropped_counts["event.retry_exhausted"] == 2

        if reporter._finalizer is not None:
            reporter._finalizer.detach()
        reporter._close_completed = True
        await reporter._http.aclose()


@pytest.mark.unit
def test_fork_reset_replaces_fold_lock_and_discards_inherited_aggregate() -> None:
    reporter = _quiet_sync()
    reporter._fold_or_count_event_drop(_event(), "retry_exhausted")
    reporter._queue.append(_PendingEvent(_event(call_id=call_uuid("fork-token"))))
    claim = reporter._pop_batch_in_hand(now=0.0, final=True)
    assert claim is not None
    settlement_event = _event(call_id=call_uuid("fork-settlement-token"))
    reporter.report_settlement(_confirm(call_id=settlement_event.call_id), settlement_event)
    settlement_claim = reporter._pop_settlement_in_hand()
    assert settlement_claim is not None
    old_lock = reporter._receipt_fold_state.lock
    old_ownership_lock = reporter._ownership_lock
    old_http = reporter._http

    reporter._reset_after_fork_in_child()

    assert reporter._receipt_fold_state.lock is not old_lock
    assert reporter._ownership_lock is not old_ownership_lock
    assert reporter._receipt_fold_snapshot() == {}
    assert reporter._event_batches_in_hand == {}
    assert reporter._next_event_batch_token == 0
    assert reporter._settlements_in_hand == {}
    assert reporter._next_settlement_token == 0
    reporter._shutdown.set()
    reporter._http.close()
    old_http.close()


@pytest.mark.unit
def test_fork_reset_recovers_inherited_mid_seal_delivery_state() -> None:
    reporter = _quiet_sync(batch_size=1)
    orphan_event = _event(call_id=call_uuid("fork-mid-seal-orphan-event"))
    reporter._queue.append(_PendingEvent(orphan_event))
    event_claim = reporter._pop_batch_in_hand(now=0.0, final=True)
    assert event_claim is not None

    orphan_settlement_event = _event(call_id=call_uuid("fork-mid-seal-orphan-settlement"))
    reporter.report_settlement(
        _confirm(call_id=orphan_settlement_event.call_id), orphan_settlement_event
    )
    settlement_claim = reporter._pop_settlement_in_hand()
    assert settlement_claim is not None

    reporter._fold_or_count_event_drop(
        _event(call_id=call_uuid("fork-mid-seal-fold"), receipt_aggregate_count=7),
        "retry_exhausted",
    )
    retained = _event(
        call_id=call_uuid("fork-mid-seal-retained"),
        receipt_aggregate_count=11,
        input_tokens=55,
        estimated_output_bound=77,
    )
    reporter._queue.append(_PendingEvent(retained))
    reporter._shutdown.set()
    with reporter._ownership_lock:
        # Exact child snapshot after the seal installs its refusal gate but
        # before it claims queues/tokens/folds and publishes completion.
        reporter._delivery_closed = True
        reporter._delivery_completed = False

    old_http = reporter._http
    old_ownership_lock = reporter._ownership_lock
    sent: list[dict[str, object]] = []

    reporter._reset_after_fork_in_child()

    assert reporter._ownership_lock is not old_ownership_lock
    assert reporter._delivery_closed is False
    assert reporter._delivery_completed is False
    assert reporter._event_batches_in_hand == {}
    assert reporter._settlements_in_hand == {}
    assert reporter._receipt_fold_snapshot() == {}
    assert [item.event.call_id for item in reporter._queue] == [retained.call_id]

    def healthy(_url: str, **kwargs: object) -> MagicMock:
        sent.extend(kwargs["json"])  # type: ignore[arg-type, index]
        return _ok_response()

    with patch.object(reporter._http, "post", side_effect=healthy):
        reporter.close(timeout=1.0)

    assert [payload["call_id"] for payload in sent] == [retained.call_id]
    assert sent[0]["receipt_aggregate_count"] == 11
    assert not reporter._queue
    assert reporter._receipt_fold_snapshot() == {}
    assert reporter.dropped_counts == {}
    assert reporter._delivery_closed is True
    assert reporter._delivery_completed is True
    assert reporter._close_is_completed() is True
    old_http.close()


@pytest.mark.unit
def test_fork_reset_does_not_reopen_terminal_fold_ownership() -> None:
    reporter = _quiet_sync()
    reporter._receipt_fold_state.take_for_cycle(final=True)
    old_http = reporter._http

    reporter._reset_after_fork_in_child()
    reporter._fold_or_count_event_drop(_event(receipt_aggregate_count=4), "closed_enqueue")

    assert reporter._receipt_fold_snapshot() == {}
    assert reporter.dropped_counts == {"event.closed_enqueue": 4}
    reporter._shutdown.set()
    reporter._http.close()
    old_http.close()
