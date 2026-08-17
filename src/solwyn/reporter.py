"""Async metadata and provider-health reporter.

MetadataReporter (sync, background thread queue) and AsyncMetadataReporter
(asyncio.create_task) batch and flush metadata events plus current circuit-breaker
snapshots to the Solwyn cloud API. Neither blocks the LLM call path.

Delivery is at-least-once: confirms, settlements, and metadata batches are
retried with bounded backoff, a control-plane-breaker-open flush HOLDS work
instead of dropping it. Unavoidable denial-receipt losses fold into bounded
content-free aggregates for recovery replay; every other drop is counted
(``dropped_counts``) and logged at a bounded rate. The server dedups duplicate
sends via an idempotency ledger, so retries (and fork/shutdown races) are safe.

Events contain cost/latency metadata only -- never prompts or responses.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import logging
import math
import re
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Literal, NamedTuple, TypeVar

import httpx

from solwyn import _base
from solwyn._constants import ORDINARY_TOKEN_COUNT_MAX
from solwyn._lifecycle import (
    _drain_count,
    _is_retryable_exc,
    register_async_reporter,
    register_fork_reset,
    register_sync_reporter,
)
from solwyn._read_only_key import handle_read_only_key_error
from solwyn._surfaces import SurfaceContext
from solwyn._types import (
    BreakerStateReport,
    BudgetConfirmRequest,
    CallStatus,
    CircuitState,
    MediaUsage,
    MetadataEvent,
    Modality,
    ProviderName,
    UntrackedSurfaceReport,
)
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerState

logger = logging.getLogger(__name__)


def _log_warning(message: str, *args: object) -> None:
    """Invoke host logging without allowing a handler to break delivery."""
    with suppress(Exception):
        logger.warning(message, *args)


def _log_error(message: str, *args: object) -> None:
    with suppress(Exception):
        logger.error(message, *args)


def _log_debug(message: str, *args: object) -> None:
    with suppress(Exception):
        logger.debug(message, *args)


# Patchable clock seam: all backoff/deadline arithmetic reads through this so
# tests can drive time deterministically.
_monotonic = time.monotonic

# At most one aggregated drop WARNING per this many seconds (the first drop ever
# logs immediately).
_DROP_LOG_INTERVAL = 60.0

# Bound for the standalone confirm + settlement queues. Distinct from the
# (larger) event queue bound so a control-plane outage cannot let pending
# settlements grow without limit.
_MAX_PENDING_CONTROL = 1000

# Untracked-surface reports are advisory deltas, not spend events. A key is
# eligible on its first observation and then no more than once per 15 minutes.
_UNTRACKED_REPORT_INTERVAL = 15 * 60.0
_UNTRACKED_REPORT_BATCH_SIZE = 100
_UNTRACKED_OCCURRENCES_MAX = 1_000_000_000
_UNTRACKED_REPORT_KEY_LIMIT = 512

# A denial receipt that cannot be delivered is compressed into this many
# per-(run, source) aggregates. Existing keys always remain writable at the
# bound; a new key is observably refused rather than evicting another run's
# durable trace.
_RECEIPT_FOLD_LIMIT = 256

_T = TypeVar("_T")

# Raw C0/DEL/C1 control bytes. A compliant server repr-escapes these before
# they reach the wire, so the substitution below is a no-op on compliant
# bodies and verbatim logging is preserved.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _escape_control(value: str) -> str:
    """Repr-escape raw control bytes in a server-echoed value.

    Defense-in-depth: the server should have escaped these already — a
    server-side escaping regression (or a misrouted/compromised endpoint)
    must not be able to inject forged log lines or ANSI sequences into
    customer logs via echoed model names or rejection messages.
    """
    return _CONTROL_CHARS.sub(lambda m: repr(m.group())[1:-1], value)


class _SendOutcome(enum.Enum):
    """The disposition of one send attempt."""

    SENT = "sent"  # delivered, or a read-only-key policy skip (terminal, not a loss)
    HELD = "held"  # control-plane breaker refused admission — not an attempt
    RETRY = "retry"  # transient failure — retry after backoff
    DROPPED = "dropped"  # terminal status / poison item — never retry


class _IngestRejectionKind(enum.Enum):
    """How precisely a successful ingest response identifies rejections."""

    CLEAN = "clean"
    EXACT = "exact"
    LEGACY = "legacy"
    MALFORMED = "malformed"


class _CloseFinalizationClaim(enum.Enum):
    """Why one sync close caller did or did not acquire whole-close ownership."""

    ACQUIRED = "acquired"
    COMPLETED = "completed"
    REENTRANT = "reentrant"
    BUSY = "busy"
    EXPIRED = "expired"


@dataclasses.dataclass(frozen=True)
class _IngestRejections:
    """Side-effect-free parsing result for one successful ingest response."""

    kind: _IngestRejectionKind
    indexes: frozenset[int] = frozenset()
    count: int = 0


@dataclasses.dataclass(frozen=True)
class _BatchSendResult:
    """Transport result plus any terminal members in an accepted batch."""

    outcome: _SendOutcome
    rejections: _IngestRejections


@dataclasses.dataclass
class _PendingConfirm:
    """A standalone confirm awaiting delivery, with retry bookkeeping."""

    request: BudgetConfirmRequest
    attempts: int = 0
    next_attempt_at: float = 0.0


@dataclasses.dataclass
class _PendingEvent:
    """A metadata event awaiting delivery, with retry bookkeeping."""

    event: MetadataEvent
    attempts: int = 0
    next_attempt_at: float = 0.0


@dataclasses.dataclass(frozen=True)
class _EventBatchToken:
    """Opaque identity for one sync drain's claimed event prefix."""

    value: int


@dataclasses.dataclass(frozen=True)
class _ClaimedEventBatch:
    """A sync event prefix whose ownership is registered by token."""

    token: _EventBatchToken
    pending: tuple[_PendingEvent, ...]


@dataclasses.dataclass(frozen=True)
class _SettlementToken:
    """Opaque identity for one sync drain's claimed settlement pair."""

    value: int


@dataclasses.dataclass
class _PendingSettlement:
    """A reservation settlement: its confirm (retryable) paired with its event."""

    confirm: _PendingConfirm
    event: MetadataEvent


@dataclasses.dataclass(frozen=True)
class _ClaimedSettlement:
    """A sync settlement pair whose ownership is registered by token."""

    token: _SettlementToken
    pending: _PendingSettlement


class _ReceiptFoldKey(NamedTuple):
    """Pricing-compatible, content-free identity for one receipt aggregate."""

    run_id: str
    original_source: str
    model: str
    provider: ProviderName
    provider_region: str | None
    service_tier: str | None
    modality: Modality
    receipt_pricing_input_tokens: int | None
    has_media_usage: bool
    has_image_count: bool
    has_generation_count: bool
    has_video_seconds: bool
    has_audio_seconds: bool
    has_input_characters: bool
    media_resolution: str | None
    media_quality: str | None
    media_is_estimated: bool


@dataclasses.dataclass
class _ReceiptFold:
    """Content-free structural aggregate for denied events lost in transit.

    Model/provider are first-writer-wins for deterministic replay. Source
    timestamps are retained for local diagnostics even though replay events use
    a fresh timestamp as required by the wire contract.
    """

    count: int
    input_tokens: int
    output_tokens: int
    estimated_output_bound: int
    image_count: int | None
    generation_count: int | None
    video_seconds: float | None
    audio_seconds: float | None
    input_characters: int | None
    first_timestamp: datetime
    last_timestamp: datetime
    model: str
    provider: ProviderName


class _ReceiptFoldState:
    """Bounded, fork-repairable fold table independent of reporter transport."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.folds: dict[_ReceiptFoldKey, _ReceiptFold] = {}
        self.previous_cycle_succeeded = False
        self.terminal = False

    def reset_after_fork_in_child(self) -> None:
        """Replace the inherited lock and avoid replaying a parent's aggregate.

        Unlike queued events, folds have no stable per-original call ids for
        server deduplication: parent and child mint distinct replay ids. Keeping
        the inherited table would therefore double-count every pre-fork fold.
        """
        terminal = self.terminal
        self.lock = threading.Lock()
        self.folds = {}
        self.previous_cycle_succeeded = False
        # Terminal ownership is one-way. A child of an already closing/closed
        # reporter has no sender either and must count late receipts instead of
        # reopening an ownerless fold table.
        self.terminal = terminal

    @staticmethod
    def _timestamp(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.utcoffset() is None else value

    def fold(self, event: MetadataEvent) -> Literal["not_denied", "folded", "overflow", "terminal"]:
        """Fold one denied event; preserve aggregate replay totals exactly."""
        if event.status != CallStatus.BUDGET_DENIED:
            return "not_denied"
        media = event.media_usage
        input_tokens = max(0, int(event.input_tokens))
        key = _ReceiptFoldKey(
            run_id=event.agent_run_id or "",
            original_source=event.deny_source or "",
            model=event.model,
            provider=event.provider,
            provider_region=event.provider_region,
            service_tier=event.service_tier,
            modality=event.modality,
            receipt_pricing_input_tokens=(
                event.receipt_pricing_input_tokens
                if event.receipt_aggregate_count is not None
                else input_tokens
            ),
            has_media_usage=media is not None,
            has_image_count=media is not None and media.image_count is not None,
            has_generation_count=media is not None and media.generation_count is not None,
            has_video_seconds=media is not None and media.video_seconds is not None,
            has_audio_seconds=media is not None and media.audio_seconds is not None,
            has_input_characters=media is not None and media.input_characters is not None,
            media_resolution=media.resolution if media is not None else None,
            media_quality=media.quality if media is not None else None,
            media_is_estimated=media.is_estimated if media is not None else False,
        )
        count = event.receipt_aggregate_count or 1
        output_tokens = max(0, int(event.output_tokens))
        output_bound = max(0, int(event.estimated_output_bound or 0))
        timestamp = self._timestamp(event.timestamp)
        with self.lock:
            if self.terminal:
                return "terminal"
            existing = self.folds.get(key)
            if existing is None:
                if len(self.folds) >= _RECEIPT_FOLD_LIMIT:
                    return "overflow"
                self.folds[key] = _ReceiptFold(
                    count=count,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_output_bound=output_bound,
                    image_count=media.image_count if media is not None else None,
                    generation_count=media.generation_count if media is not None else None,
                    video_seconds=media.video_seconds if media is not None else None,
                    audio_seconds=media.audio_seconds if media is not None else None,
                    input_characters=media.input_characters if media is not None else None,
                    first_timestamp=timestamp,
                    last_timestamp=timestamp,
                    model=event.model,
                    provider=event.provider,
                )
            else:
                existing.count += count
                existing.input_tokens += input_tokens
                existing.output_tokens += output_tokens
                existing.estimated_output_bound += output_bound
                if media is not None:
                    if media.image_count is not None:
                        existing.image_count = (existing.image_count or 0) + media.image_count
                    if media.generation_count is not None:
                        existing.generation_count = (
                            existing.generation_count or 0
                        ) + media.generation_count
                    if media.video_seconds is not None:
                        existing.video_seconds = (
                            existing.video_seconds or 0.0
                        ) + media.video_seconds
                    if media.audio_seconds is not None:
                        existing.audio_seconds = (
                            existing.audio_seconds or 0.0
                        ) + media.audio_seconds
                    if media.input_characters is not None:
                        existing.input_characters = (
                            existing.input_characters or 0
                        ) + media.input_characters
                existing.first_timestamp = min(existing.first_timestamp, timestamp)
                existing.last_timestamp = max(existing.last_timestamp, timestamp)
        return "folded"

    def snapshot(self) -> dict[_ReceiptFoldKey, _ReceiptFold]:
        with self.lock:
            return {key: dataclasses.replace(fold) for key, fold in self.folds.items()}

    def take_for_cycle(self, *, final: bool) -> list[tuple[_ReceiptFoldKey, _ReceiptFold]]:
        """Atomically consume folds when recovery was proven, or once at close."""
        with self.lock:
            if final:
                # Final ownership is a one-way transition. Any replay that is
                # later lost is counted immediately; retaining it here would
                # create state with no future sender after close/GC/atexit.
                self.terminal = True
            should_drain = final or self.previous_cycle_succeeded
            self.previous_cycle_succeeded = False
            if not should_drain or not self.folds:
                return []
            folds = [(key, dataclasses.replace(fold)) for key, fold in self.folds.items()]
            self.folds.clear()
            return folds

    def note_cycle_success(self) -> None:
        with self.lock:
            if not self.terminal:
                self.previous_cycle_succeeded = True


def _build_receipt_replay_event(
    key: _ReceiptFoldKey, fold: _ReceiptFold, sdk_instance_id: str | None
) -> MetadataEvent:
    """Build a fresh, content-free aggregate for the normal ingest path."""
    media_usage = None
    if key.has_media_usage:
        media_usage = MediaUsage(
            image_count=fold.image_count,
            generation_count=fold.generation_count,
            video_seconds=fold.video_seconds,
            audio_seconds=fold.audio_seconds,
            input_characters=fold.input_characters,
            resolution=key.media_resolution,
            quality=key.media_quality,
            is_estimated=key.media_is_estimated,
        )
    return MetadataEvent(
        model=key.model,
        provider=key.provider,
        provider_region=key.provider_region,
        service_tier=key.service_tier,
        modality=key.modality,
        input_tokens=fold.input_tokens,
        output_tokens=fold.output_tokens,
        token_details=None,
        media_usage=media_usage,
        latency_ms=0.0,
        status=CallStatus.BUDGET_DENIED,
        is_model_fallback=False,
        call_id=str(uuid.uuid4()),
        sdk_instance_id=sdk_instance_id or "receipt-fold",
        timestamp=datetime.now(UTC),
        agent_run_id=key.run_id or None,
        deny_source="aggregate_replay",
        estimated_output_bound=fold.estimated_output_bound,
        receipt_aggregate_count=fold.count,
        receipt_pricing_input_tokens=key.receipt_pricing_input_tokens,
    )


def _split_receipt_fold(fold: _ReceiptFold) -> list[_ReceiptFold]:
    """Partition one unbounded in-memory sum into 100M wire-safe folds."""

    def chunks_needed(value: int | float | None) -> int:
        if value is None:
            return 1
        if isinstance(value, int):
            return max(
                1,
                (value + ORDINARY_TOKEN_COUNT_MAX - 1) // ORDINARY_TOKEN_COUNT_MAX,
            )
        return max(1, math.ceil(value / ORDINARY_TOKEN_COUNT_MAX))

    chunk_count = max(
        chunks_needed(fold.count),
        chunks_needed(fold.input_tokens),
        chunks_needed(fold.output_tokens),
        chunks_needed(fold.estimated_output_bound),
        chunks_needed(fold.image_count),
        chunks_needed(fold.generation_count),
        chunks_needed(fold.video_seconds),
        chunks_needed(fold.audio_seconds),
        chunks_needed(fold.input_characters),
    )
    if chunk_count == 1:
        return [dataclasses.replace(fold)]

    remaining_count = fold.count
    remaining_input = fold.input_tokens
    remaining_output = fold.output_tokens
    remaining_bound = fold.estimated_output_bound
    remaining_image_count = fold.image_count
    remaining_generation_count = fold.generation_count
    remaining_video_seconds = fold.video_seconds
    remaining_audio_seconds = fold.audio_seconds
    remaining_input_characters = fold.input_characters
    chunks: list[_ReceiptFold] = []
    for index in range(chunk_count):
        chunks_left = chunk_count - index
        count = min(
            ORDINARY_TOKEN_COUNT_MAX,
            remaining_count - (chunks_left - 1),
        )
        input_tokens = min(ORDINARY_TOKEN_COUNT_MAX, remaining_input)
        output_tokens = min(ORDINARY_TOKEN_COUNT_MAX, remaining_output)
        output_bound = min(ORDINARY_TOKEN_COUNT_MAX, remaining_bound)
        image_count = (
            None
            if remaining_image_count is None
            else min(ORDINARY_TOKEN_COUNT_MAX, remaining_image_count)
        )
        generation_count = (
            None
            if remaining_generation_count is None
            else min(ORDINARY_TOKEN_COUNT_MAX, remaining_generation_count)
        )
        video_seconds = (
            None
            if remaining_video_seconds is None
            else min(float(ORDINARY_TOKEN_COUNT_MAX), remaining_video_seconds)
        )
        audio_seconds = (
            None
            if remaining_audio_seconds is None
            else min(float(ORDINARY_TOKEN_COUNT_MAX), remaining_audio_seconds)
        )
        input_characters = (
            None
            if remaining_input_characters is None
            else min(ORDINARY_TOKEN_COUNT_MAX, remaining_input_characters)
        )
        chunks.append(
            dataclasses.replace(
                fold,
                count=count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_output_bound=output_bound,
                image_count=image_count,
                generation_count=generation_count,
                video_seconds=video_seconds,
                audio_seconds=audio_seconds,
                input_characters=input_characters,
            )
        )
        remaining_count -= count
        remaining_input -= input_tokens
        remaining_output -= output_tokens
        remaining_bound -= output_bound
        if remaining_image_count is not None and image_count is not None:
            remaining_image_count -= image_count
        if remaining_generation_count is not None and generation_count is not None:
            remaining_generation_count -= generation_count
        if remaining_video_seconds is not None and video_seconds is not None:
            remaining_video_seconds -= video_seconds
        if remaining_audio_seconds is not None and audio_seconds is not None:
            remaining_audio_seconds -= audio_seconds
        if remaining_input_characters is not None and input_characters is not None:
            remaining_input_characters -= input_characters
    return chunks


def _build_receipt_replay_events(
    key: _ReceiptFoldKey,
    fold: _ReceiptFold,
    sdk_instance_id: str | None,
) -> list[MetadataEvent]:
    """Build fresh replay identities for every 100M-safe partition."""
    return [
        _build_receipt_replay_event(key, chunk, sdk_instance_id)
        for chunk in _split_receipt_fold(fold)
    ]


_UntrackedSurfaceKey = tuple[str, str, str, str]


@dataclasses.dataclass(frozen=True)
class _BuiltUntrackedReport:
    """One wire report plus the cumulative count it consumes on a 2xx."""

    generation: int
    key: _UntrackedSurfaceKey
    sent_through_occurrences: int
    report: UntrackedSurfaceReport


class _UntrackedReportState:
    """Reporter-owned advisory state safe to capture in lifecycle finalizers.

    The holder has no reporter or transport reference, so the async GC
    finalizer can retain it without keeping the reporter alive. Its lock is
    replaceable after fork while the holder identity stays stable.
    """

    def __init__(self, sdk_instance_id: str | None) -> None:
        self.sdk_instance_id = sdk_instance_id
        self.lock = threading.Lock()
        self.generation = 0
        self.observations: dict[_UntrackedSurfaceKey, _base._UntrackedSurfaceObservation] = {}
        self.last_sent_occurrences: dict[_UntrackedSurfaceKey, int] = {}
        self.last_attempted_at: dict[_UntrackedSurfaceKey, float] = {}

    def reset_after_fork_in_child(self) -> None:
        """Discard inherited eligibility and invalidate captured report batches."""
        self.lock = threading.Lock()
        self.generation += 1
        self.observations = {}
        self.last_sent_occurrences = {}
        self.last_attempted_at = {}

    def observe(
        self,
        *,
        context: SurfaceContext,
        surface: str,
        rule_kind: Literal["unmetered_spend", "unknown"],
        capability_scope: str | None,
        posture: Literal["warn", "allow"],
        seen_at: datetime,
    ) -> None:
        key = (context.provider, context.client_shape, context.mode, surface)
        with self.lock:
            observation = self.observations.get(key)
            if observation is None:
                if len(self.observations) >= _UNTRACKED_REPORT_KEY_LIMIT:
                    return
                self.observations[key] = {
                    "occurrences": 1,
                    "first_seen_at": seen_at,
                    "last_seen_at": seen_at,
                    "rule_kind": rule_kind,
                    "capability_scope": capability_scope,
                    "posture": posture,
                    "warning_emitted": False,
                }
                return
            observation["occurrences"] += 1
            observation["first_seen_at"] = min(observation["first_seen_at"], seen_at)
            observation["last_seen_at"] = max(observation["last_seen_at"], seen_at)
            observation["rule_kind"] = rule_kind
            observation["capability_scope"] = capability_scope
            observation["posture"] = posture

    def build_reports(self, now: float) -> list[_BuiltUntrackedReport]:
        """Build due deltas without advancing successful-send state."""
        if self.sdk_instance_id is None:
            return []
        with self.lock:
            observations = [
                (key, observation.copy()) for key, observation in self.observations.items()
            ]
            last_occurrences = dict(self.last_sent_occurrences)
            last_attempted_at = dict(self.last_attempted_at)
            generation = self.generation

        reports: list[_BuiltUntrackedReport] = []
        for key, observation in sorted(observations, key=lambda item: item[0]):
            provider, client_shape, mode, surface = key
            baseline = last_occurrences.get(key, 0)
            total = observation["occurrences"]
            if total <= baseline:
                continue
            previous_attempt_at = last_attempted_at.get(key)
            if (
                previous_attempt_at is not None
                and now < previous_attempt_at + _UNTRACKED_REPORT_INTERVAL
            ):
                continue
            delta = min(total - baseline, _UNTRACKED_OCCURRENCES_MAX)
            try:
                report = UntrackedSurfaceReport.model_validate(
                    {
                        "provider": provider,
                        "client_shape": client_shape,
                        "mode": mode,
                        "surface": surface,
                        "rule_kind": observation["rule_kind"],
                        "capability_scope": observation["capability_scope"],
                        "posture": observation["posture"],
                        "occurrences": delta,
                        "first_seen_at": observation["first_seen_at"],
                        "last_seen_at": observation["last_seen_at"],
                        "sdk_instance_id": self.sdk_instance_id,
                        "report_id": str(uuid.uuid4()),
                    }
                )
            except Exception:
                # S1 validates each structural field before observation. A
                # corrupted private-test entry remains advisory and silent,
                # but it must still advance cadence: leaving the key due would
                # make both worker completion hooks respawn without a delay.
                with self.lock:
                    if generation == self.generation:
                        attempted_at = self.last_attempted_at.get(key)
                        if attempted_at is None or now > attempted_at:
                            self.last_attempted_at[key] = now
                continue
            reports.append(
                _BuiltUntrackedReport(
                    generation=generation,
                    key=key,
                    sent_through_occurrences=baseline + delta,
                    report=report,
                )
            )
        return reports

    def reports_due(self, now: float) -> bool:
        if self.sdk_instance_id is None:
            return False
        with self.lock:
            for key, observation in self.observations.items():
                if observation["occurrences"] <= self.last_sent_occurrences.get(key, 0):
                    continue
                attempted_at = self.last_attempted_at.get(key)
                if attempted_at is None or now >= attempted_at + _UNTRACKED_REPORT_INTERVAL:
                    return True
        return False

    def mark_attempted(self, reports: list[_BuiltUntrackedReport], attempted_at: float) -> None:
        with self.lock:
            for built in reports:
                if built.generation != self.generation:
                    continue
                self.last_attempted_at[built.key] = attempted_at

    def mark_sent(self, reports: list[_BuiltUntrackedReport]) -> None:
        with self.lock:
            for built in reports:
                if built.generation != self.generation:
                    continue
                current = self.last_sent_occurrences.get(built.key, 0)
                self.last_sent_occurrences[built.key] = max(current, built.sent_through_occurrences)


class _ReporterBase:
    """Sans-I/O base class for metadata reporting.

    Manages the pending queues, retry/backoff arithmetic, and drop accounting.
    Subclasses add the I/O layer (threading or asyncio) and HTTP transport.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 10_000,
        max_in_flight: int = 3,
        breaker_snapshots: Callable[[], list[tuple[ProviderName, CircuitBreakerState]]]
        | None = None,
        sdk_instance_id: str | None = None,
        breaker_reporting_enabled: bool = True,
        report_untracked_surfaces: bool = True,
        breaker_report_heartbeat: float = 60.0,
        control_plane_breaker: CircuitBreaker | None = None,
        max_send_attempts: int = 5,
        retry_backoff_base: float = 1.0,
        retry_backoff_cap: float = 60.0,
        shutdown_deadline: float = 5.0,
    ) -> None:
        if max_queue_size < 1:
            # A zero-capacity queue has no defined drop-oldest semantics: the
            # sync bound would evict every append (all spend counted dropped)
            # while the async bound retained one item — reject it up front.
            raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size}")
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        self.max_in_flight = max_in_flight
        # At-least-once delivery bounds (see SolwynConfig.reporter_*): retries
        # per item before a counted drop, exponential backoff base/cap, and the
        # single wall-clock budget the shutdown/exit flush chain may spend.
        self.max_send_attempts = max_send_attempts
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_cap = retry_backoff_cap
        self.shutdown_deadline = shutdown_deadline
        self._breaker_snapshots = breaker_snapshots
        self._sdk_instance_id = sdk_instance_id
        self._breaker_reporting_enabled = breaker_reporting_enabled
        self._breaker_report_heartbeat = breaker_report_heartbeat
        # Shared with the budget enforcer's check path: a streak of confirm
        # failures opens this breaker so a known-down confirm is HELD (not
        # dropped) without paying the timeout. Never a provider — excluded from
        # _build_breaker_reports (that is provider health only).
        self._control_plane_breaker = control_plane_breaker
        self._breaker_project_id: str | None = None
        self._breaker_project_lock = threading.Lock()
        self._breaker_report_lock = threading.Lock()
        # provider -> (state, failure_count, success_count) of the last report
        # that got a 2xx. Written on successful POST only, so a failed send
        # keeps differing from its snapshot and self-retries next cycle.
        self._breaker_last_sent: dict[str, tuple[CircuitState, int, int]] = {}
        self._breaker_heartbeat_at = 0.0
        self._untracked_state = (
            _UntrackedReportState(sdk_instance_id) if report_untracked_surfaces else None
        )

        # Plain deques (NO maxlen): bounds are enforced by _enqueue_owned so an
        # overflow is COUNTED, not silently dropped by the deque itself. The
        # confirm/settlement queues hold fire-and-forgotten spend settlements so
        # the user's thread is never blocked on an httpx.post to Solwyn.
        self._queue: collections.deque[_PendingEvent] = collections.deque()
        self._confirm_queue: collections.deque[_PendingConfirm] = collections.deque()
        self._settlement_queue: collections.deque[_PendingSettlement] = collections.deque()
        self._in_flight = 0
        self._consecutive_confirm_failures = 0
        self._confirm_failure_threshold = 10
        self._consecutive_unparseable_responses = 0
        self._unparseable_response_threshold = 10

        # Drop accounting. Guarded by its own lock so enqueues from non-loop
        # threads and the flush loop never race the counters.
        self._drop_counts: dict[str, int] = {}
        self._drop_lock = threading.Lock()
        self._drops_since_last_log = 0
        self._last_drop_log_at: float | None = None
        self._logged_first_drop = False

        # Denial-only durability tail. Its own lock lets lifecycle finalizers
        # retain the content-free state without retaining this reporter, and
        # keeps folding bounded even when producer threads race a flush.
        self._receipt_fold_state = _ReceiptFoldState()

        # Shutdown ownership. Every enqueue and (on the sync path) every
        # popped-but-unresolved ("in hand") drain item is tracked under this
        # lock so a completed close() can take final ownership of ALL
        # undelivered spend — a stuck flush thread must not requeue into, and a
        # racing producer must not append into, a queue nothing will ever drain.
        # See _seal_delivery. The async reporter never populates the token
        # maps or _in_hand (its in-hand accounting rides the drains'
        # CancelledError handlers) but shares the enqueue gate and the seal's
        # straggler sweep.
        self._ownership_lock = threading.Lock()
        # Standalone confirms are fungible for accounting, so a count is
        # sufficient. Event batches and settlement pairs have exact identities:
        # concurrent background/final drains may resolve out of order, and each
        # response must publish against its own opaque-token-owned prefix/pair.
        self._in_hand: dict[str, int] = {}
        self._next_event_batch_token = 0
        self._event_batches_in_hand: dict[_EventBatchToken, tuple[_PendingEvent, ...]] = {}
        self._next_settlement_token = 0
        self._settlements_in_hand: dict[_SettlementToken, _PendingSettlement] = {}
        self._delivery_closed = False
        # Final replay is bounded by _RECEIPT_FOLD_LIMIT, but can legitimately
        # exceed the ordinary queue capacity. Once final ownership begins the
        # event path bypasses its normal producer bound until seal.
        self._final_delivery_started = False

    def observe_project_id(self, project_id: str | None) -> None:
        """Remember the non-empty project resolved by a real/cached budget check."""
        if not project_id:
            return
        with self._breaker_project_lock:
            self._breaker_project_id = project_id

    def _breaker_reporting_project_id(self) -> str | None:
        """Return the learned project only when breaker reporting can run."""
        if (
            not self._breaker_reporting_enabled
            or self._breaker_snapshots is None
            or self._sdk_instance_id is None
        ):
            return None
        with self._breaker_project_lock:
            return self._breaker_project_id

    def _build_breaker_reports(
        self, *, force: bool = False
    ) -> list[tuple[str, BreakerStateReport]]:
        """Eagerly build one timestamped current-state report cycle."""
        reports: list[tuple[str, BreakerStateReport]] = []
        project_id = self._breaker_reporting_project_id()
        if project_id is None or self._breaker_snapshots is None or self._sdk_instance_id is None:
            return reports
        try:
            snapshots = self._breaker_snapshots()
            reported_at = datetime.now(UTC)
        except Exception as exc:
            _log_warning(
                "reporter.breaker_snapshot_failed: exc_type=%s",
                type(exc).__name__,
            )
            return reports
        now = time.monotonic()
        with self._breaker_report_lock:
            heartbeat_due = now - self._breaker_heartbeat_at >= self._breaker_report_heartbeat
            if force or heartbeat_due:
                self._breaker_heartbeat_at = now
            last_sent = dict(self._breaker_last_sent)
        for provider, snapshot in snapshots:
            try:
                key = (snapshot.state, snapshot.failure_count, snapshot.success_count)
                if not force and not heartbeat_due and last_sent.get(provider.value) == key:
                    continue
                reports.append(
                    (
                        project_id,
                        BreakerStateReport(
                            provider=provider,
                            state=snapshot.state,
                            failure_count=snapshot.failure_count,
                            success_count=snapshot.success_count,
                            reported_at=reported_at,
                            sdk_instance_id=self._sdk_instance_id,
                        ),
                    ),
                )
            except Exception as exc:
                _log_warning(
                    "reporter.breaker_snapshot_invalid: provider=%s exc_type=%s",
                    provider.value,
                    type(exc).__name__,
                )
        return reports

    def _breaker_reports_due(self) -> bool:
        """Cheap pre-check so idle flush ticks never spawn a breaker cycle."""
        if self._breaker_reporting_project_id() is None or self._breaker_snapshots is None:
            return False
        with self._breaker_report_lock:
            if time.monotonic() - self._breaker_heartbeat_at >= self._breaker_report_heartbeat:
                return True
            last_sent = dict(self._breaker_last_sent)
        try:
            snapshots = self._breaker_snapshots()
        except Exception:
            return False
        return any(
            last_sent.get(provider.value)
            != (snapshot.state, snapshot.failure_count, snapshot.success_count)
            for provider, snapshot in snapshots
        )

    def observe_untracked_surface(
        self,
        *,
        context: SurfaceContext,
        surface: str,
        rule_kind: Literal["unmetered_spend", "unknown"],
        capability_scope: str | None,
        posture: Literal["warn", "allow"],
        seen_at: datetime,
    ) -> None:
        """Record one content-free observation owned by this reporter instance."""
        state = self._untracked_state
        if state is None:
            return
        state.observe(
            context=context,
            surface=surface,
            rule_kind=rule_kind,
            capability_scope=capability_scope,
            posture=posture,
            seen_at=seen_at,
        )
        self._notify_untracked_observation()

    def _notify_untracked_observation(self) -> None:
        """Let the concrete reporter schedule advisory work off the caller thread."""

    def _build_untracked_reports(self) -> list[_BuiltUntrackedReport]:
        """Build eligible per-key deltas without advancing successful-send state."""
        state = self._untracked_state
        return [] if state is None else state.build_reports(_monotonic())

    def _untracked_reports_due(self) -> bool:
        """Cheap cadence/delta pre-check for the reporter flush loop."""
        state = self._untracked_state
        return state is not None and state.reports_due(_monotonic())

    def _mark_untracked_reports_attempted(self, reports: list[_BuiltUntrackedReport]) -> None:
        """Advance cadence for every network attempt, successful or otherwise."""
        state = self._untracked_state
        if state is not None:
            state.mark_attempted(reports, _monotonic())

    def _mark_untracked_reports_sent(self, reports: list[_BuiltUntrackedReport]) -> None:
        """Advance delta baselines only after a successful POST."""
        state = self._untracked_state
        if state is not None:
            state.mark_sent(reports)

    # ------------------------------------------------------------------
    # Queueing + bounds
    # ------------------------------------------------------------------

    def _receipt_fold_snapshot(self) -> dict[_ReceiptFoldKey, _ReceiptFold]:
        """Return an isolated diagnostic snapshot for tests and observability."""
        return self._receipt_fold_state.snapshot()

    @staticmethod
    def _event_drop_weight(event: MetadataEvent) -> int:
        """Underlying receipt count represented by a terminal event loss."""
        if event.status == CallStatus.BUDGET_DENIED:
            return event.receipt_aggregate_count or 1
        return 1

    def _fold_or_record_event_drop(self, event: MetadataEvent, reason: str) -> None:
        """Mutate fold/drop state without invoking logging callbacks."""
        outcome = self._receipt_fold_state.fold(event)
        if outcome == "not_denied":
            self._record_drop("event", reason)
        elif outcome == "overflow":
            self._record_drop(
                "event",
                "receipt_fold_overflow",
                n=self._event_drop_weight(event),
            )
        elif outcome == "terminal":
            self._record_drop("event", reason, n=self._event_drop_weight(event))

    def _fold_or_count_event_drop(self, event: MetadataEvent, reason: str) -> None:
        """Fold denied receipts; preserve ordinary event drop accounting."""
        self._fold_or_record_event_drop(event, reason)
        self._maybe_log_drops()

    def _dispose_pending_events(
        self, pending: list[_PendingEvent], reason: str, *, emit: bool = True
    ) -> None:
        for item in pending:
            self._fold_or_record_event_drop(item.event, reason)
        if emit:
            self._maybe_log_drops()

    def _publish_ingest_rejections(
        self,
        events: list[MetadataEvent],
        rejections: _IngestRejections,
        *,
        emit: bool = True,
    ) -> None:
        """Publish terminal 202 members after their transport owner resolves.

        Exact identities take the normal denial-aware disposition. A partial
        legacy response preserves only its proven count because selecting an
        event would guess at identity; a full-batch legacy rejection proves
        every identity and can use the same denial-aware disposition.
        Malformed bodies retain the existing fail-open behavior.
        """
        if rejections.kind is _IngestRejectionKind.EXACT:
            for index in sorted(rejections.indexes):
                if 0 <= index < len(events):
                    self._fold_or_record_event_drop(events[index], "ingest_rejected")
        elif rejections.kind is _IngestRejectionKind.LEGACY:
            if rejections.count == len(events):
                # Index-less responses normally prove only a count. When the
                # server rejected the ENTIRE batch, however, every identity is
                # proven terminal: dispose each event so an aggregate replay
                # preserves its represented receipt cardinality and sums.
                for event in events:
                    self._fold_or_record_event_drop(event, "ingest_rejected")
            else:
                self._record_drop("event", "ingest_rejected", n=min(rejections.count, len(events)))
        if emit:
            self._maybe_log_drops()

    def _dispose_settlements(
        self, pending: list[_PendingSettlement], reason: str, *, emit: bool = True
    ) -> None:
        if not pending:
            return
        self._record_drop("settlement_confirm", reason, n=len(pending))
        for item in pending:
            self._fold_or_record_event_drop(item.event, reason)
        if emit:
            self._maybe_log_drops()

    def _enqueue_event_locked(self, event: MetadataEvent) -> None:
        """Append one event and own any bound eviction without reacquiring.

        The caller holds ``_ownership_lock`` and has already established that
        delivery is open. Publishing an overflow disposition before releasing
        that lock keeps a settlement's token-to-event transition indivisible
        from close/seal.
        """
        self._queue.append(_PendingEvent(event))
        if self._final_delivery_started or len(self._queue) <= self.max_queue_size:
            return
        try:
            evicted = self._queue.popleft()
        except IndexError:
            return
        self._fold_or_record_event_drop(evicted.event, "overflow")

    def _enqueue_event_owned(self, event: MetadataEvent) -> bool:
        with self._ownership_lock:
            if self._delivery_closed:
                return False
            self._enqueue_event_locked(event)
        self._maybe_log_drops()
        return True

    def _replay_event(self, key: _ReceiptFoldKey, fold: _ReceiptFold) -> MetadataEvent:
        """Build one fresh normal-wire aggregate from content-free fold state."""
        return _build_receipt_replay_event(key, fold, self._sdk_instance_id)

    def _drain_receipt_folds_to_queue(self, *, final: bool) -> None:
        """Move one eligible fold snapshot through the ordinary event queue."""
        with self._ownership_lock:
            folds = self._receipt_fold_state.take_for_cycle(final=final)
            replay_events = [
                self._replay_event(key, chunk)
                for key, fold in folds
                for chunk in _split_receipt_fold(fold)
            ]
            if final:
                self._final_delivery_started = True
            # The terminal snapshot is strictly bounded to 256 entries. Move
            # the whole snapshot under one ownership transaction and bypass
            # the ordinary producer capacity so a healthy close can deliver
            # every aggregate even when max_queue_size is tiny.
            if self._delivery_closed:
                for event in replay_events:
                    self._fold_or_record_event_drop(event, "shutdown_deadline")
            elif final:
                self._queue.extend(_PendingEvent(event) for event in replay_events)
            else:
                for event in replay_events:
                    # Non-final recovery remains subject to the ordinary
                    # capacity bound. Any eviction re-folds under this same
                    # ownership transaction, so seal can never observe a
                    # take-before-enqueue gap.
                    self._enqueue_event_locked(event)
        self._maybe_log_drops()

    def _enqueue(self, event: MetadataEvent) -> bool:
        """Add an event to the queue.  Counted drop-oldest on overflow.

        Returns False (the caller counts the refusal) once delivery has sealed.
        """
        return self._enqueue_event_owned(event)

    def _enqueue_owned(
        self,
        queue: collections.deque[_T],
        item: _T,
        maxlen: int,
        kind: str,
    ) -> bool:
        """Append under the ownership lock; refuse once delivery has sealed.

        Returns False when ``_seal_delivery`` already ran — the caller counts
        the refusal. Without this gate an enqueue could pass the shutdown
        check, lose the race with close()'s final drain, and append into a
        queue nothing will ever drain. The drop-oldest mutation and its counter
        publication share the ownership transaction; warning emission happens
        only after the lock is released.
        """
        with self._ownership_lock:
            if self._delivery_closed:
                return False
            queue.append(item)
            if len(queue) > maxlen:
                try:
                    queue.popleft()
                except IndexError:
                    pass
                else:
                    self._record_drop(kind, "overflow")
        self._maybe_log_drops()
        return True

    def _move_event_to_queue(self, event: MetadataEvent) -> None:
        """Move a settlement's metadata event onto the main event queue.

        cost_events ingest is the durable spend truth, so the event must never
        be lost because its confirm failed — it rides the normal event path.
        Once delivery has sealed nothing will drain the event queue: count the
        loss instead of appending it stranded.
        """
        enqueued = self._enqueue_event_owned(event)
        if not enqueued:
            self._fold_or_count_event_drop(event, "shutdown_deadline")

    def _enqueue_settlement_owned(self, pending: _PendingSettlement) -> bool:
        """Append a pair and atomically transfer any control-queue eviction."""
        enqueued = False
        with self._ownership_lock:
            if self._delivery_closed:
                return False
            self._settlement_queue.append(pending)
            enqueued = True
            if len(self._settlement_queue) > _MAX_PENDING_CONTROL:
                try:
                    evicted = self._settlement_queue.popleft()
                except IndexError:
                    pass
                else:
                    self._record_drop("settlement_confirm", "overflow")
                    self._enqueue_event_locked(evicted.event)
        self._maybe_log_drops()
        return enqueued

    def _mark_delivery_completed_locked(self) -> None:
        """Concrete sync reporters mark seal completion under ownership."""

    def _seal_delivery(self, *, emit: bool = True) -> bool:
        """Take final ownership of every queued or in-hand item at close().

        Runs after close()'s final flush. A join-timeout-stranded sync flush
        thread may still hold popped items mid-POST (and would otherwise
        requeue them into a dead queue), and a racing producer on any thread
        may have appended after the final drain passed its queue. Atomically:
        seal delivery (enqueues, drain pops, and requeues are refused from
        here on), claim whatever re-appeared in the queues, and count the
        flush thread's in-hand items — their owner can no longer deliver them
        within the deadline. A stuck send that later succeeds leaves a
        conservative overcount for that one item; drops are never UNDERstated.
        """
        sealed = False
        with self._ownership_lock:
            # A second sealer cannot observe this flag until the first releases
            # the same lock, by which point every mutation and disposition below
            # (including the concrete completion marker) is already published.
            if self._delivery_closed:
                return False
            self._delivery_closed = True
            in_hand = dict(self._in_hand)
            self._in_hand.clear()
            event_batches = list(self._event_batches_in_hand.values())
            self._event_batches_in_hand.clear()
            settlement_tokens = list(self._settlements_in_hand.values())
            self._settlements_in_hand.clear()
            n_confirm = _drain_count(self._confirm_queue)
            settlements: list[_PendingSettlement] = []
            while self._settlement_queue:
                settlements.append(self._settlement_queue.popleft())
            events: list[_PendingEvent] = []
            while self._queue:
                events.append(self._queue.popleft())
            # Settlement pairs keep exact token/queue ownership until both
            # halves have a published shutdown disposition. A seal racing a
            # late response can therefore never observe or create a gap.
            self._dispose_settlements(
                [*settlements, *settlement_tokens], "shutdown_deadline", emit=False
            )
            # Direct seals (tests, timeout races, lifecycle edges) are final
            # owners too. Consume and terminalize any folds not already moved
            # into the normal event path by the final flush.
            residual_folds = self._receipt_fold_state.take_for_cycle(final=True)
            for key, fold in residual_folds:
                for chunk in _split_receipt_fold(fold):
                    self._fold_or_record_event_drop(
                        self._replay_event(key, chunk),
                        "shutdown_deadline",
                    )
            # Claim and mutation are indivisible. In particular, an expired
            # daemon drain must never make an event disappear from both the
            # queue/token maps and the published terminal accounting while a
            # concurrent close returns. These helpers mutate only; warning
            # handlers run after ownership is released.
            self._record_drop(
                "confirm", "shutdown_deadline", n=n_confirm + in_hand.get("confirm", 0)
            )
            self._dispose_pending_events(events, "shutdown_deadline", emit=False)
            for batch in event_batches:
                self._dispose_pending_events(list(batch), "shutdown_deadline", emit=False)
            self._mark_delivery_completed_locked()
            sealed = True
        if emit:
            self._maybe_log_drops(force=True)
        return sealed

    def _requeue_cancelled(self, queue: collections.deque[_T], items: list[_T]) -> bool:
        """Return a cancelled drain's in-hand items to the head of their queue.

        A cancelled close() leaves the lifecycle rescue paths armed, and rescue
        can only retry what is IN the queues — counting a still-deliverable
        in-hand item as dropped would write off spend the exit hook could ship.
        A close() that COMPLETES instead seals right after the cancelled drain
        settles, so the requeued item still gets its counted disposition.
        Returns False once delivery has sealed (nothing will ever drain the
        queue again); the caller counts the drop.
        """
        with self._ownership_lock:
            if self._delivery_closed:
                return False
            queue.extendleft(reversed(items))
            return True

    # ------------------------------------------------------------------
    # Retry / backoff / deadline arithmetic
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempts: int) -> float:
        """Exponential backoff for the ``attempts``-th failed try (1-indexed).

        The exponent is clamped: ``2.0 ** 1024`` raises ``OverflowError``, so a
        large-but-valid ``max_send_attempts`` would poison the queue head with
        an uncatchable arithmetic error before ``min`` could apply the cap.
        (float MULTIPLICATION saturates to ``inf`` instead of raising, which
        ``min`` then resolves to the cap.)
        """
        return min(self.retry_backoff_cap, self.retry_backoff_base * 2.0 ** min(attempts - 1, 1023))

    def _resolve_retryable(self, *, attempts: int) -> tuple[bool, float]:
        """Return (finished, next_attempt_at) for a RETRY outcome.

        ``finished`` is True when the retry budget is spent (the item must be
        dropped); otherwise ``next_attempt_at`` is when it may be retried.
        """
        if attempts >= self.max_send_attempts:
            return True, 0.0
        return False, _monotonic() + self._backoff_delay(attempts)

    def _deadline_expired(self, deadline: float | None) -> bool:
        """Whether the shared shutdown deadline (if any) has been reached."""
        return deadline is not None and _monotonic() >= deadline

    def _send_timeout(self, deadline: float | None) -> float:
        """Per-request timeout, clamped into the remaining deadline window."""
        if deadline is None:
            return 5.0
        return max(0.05, min(5.0, deadline - _monotonic()))

    # ------------------------------------------------------------------
    # Drop accounting
    # ------------------------------------------------------------------

    def _record_drop(self, kind: str, reason: str, n: int = 1) -> None:
        """Record drops without invoking logging callbacks.

        This mutation-only primitive is safe inside ``_ownership_lock``. Drop
        warnings are emitted only after the ownership transaction completes.
        """
        if n <= 0:
            return
        with self._drop_lock:
            key = f"{kind}.{reason}"
            self._drop_counts[key] = self._drop_counts.get(key, 0) + n
            self._drops_since_last_log += n

    def _count_drop(self, kind: str, reason: str, n: int = 1) -> None:
        """Record ``n`` dropped spend items keyed ``"{kind}.{reason}"``.

        Logging is driven from HERE (rate-limited by ``_maybe_log_drops``), not
        only from flush cycles: the first drop ever must warn immediately, and
        a post-close drop has no later flush cycle to surface it.
        """
        self._record_drop(kind, reason, n)
        self._maybe_log_drops()

    def _maybe_log_drops(self, *, force: bool = False) -> None:
        """Emit an aggregated drop WARNING, rate-limited to one per interval.

        The first drop ever logs immediately; afterwards at most one aggregate
        line per ``_DROP_LOG_INTERVAL`` (or immediately when ``force`` is set,
        e.g. from ``close()``).
        """
        with self._drop_lock:
            if self._drops_since_last_log == 0:
                return
            now = _monotonic()
            first = not self._logged_first_drop
            due = (
                first
                or force
                or self._last_drop_log_at is None
                or (now - self._last_drop_log_at) >= _DROP_LOG_INTERVAL
            )
            if not due:
                return
            since = self._drops_since_last_log
            totals = dict(self._drop_counts)
            self._drops_since_last_log = 0
            self._last_drop_log_at = now
            self._logged_first_drop = True
        # Host applications may install arbitrary logging handlers. Spend
        # ownership/disposition is already complete; logging must never roll
        # it back, strand a token, or deadlock through reentrancy.
        _log_warning("reporter.spend_events_dropped: new=%d totals=%s", since, totals)

    @property
    def dropped_counts(self) -> dict[str, int]:
        """A snapshot of counted, undeliverable spend items by ``kind.reason``."""
        with self._drop_lock:
            return dict(self._drop_counts)

    # ------------------------------------------------------------------
    # Auth + failure accounting
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for cloud API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _confirm_failure_outcome(self, exc: Exception) -> _SendOutcome:
        """Map a confirm send failure to RETRY (transient) or DROPPED (terminal)."""
        if _is_retryable_exc(exc):
            return _SendOutcome.RETRY
        if isinstance(exc, httpx.HTTPStatusError):
            # Status code only — never a body (privacy).
            _log_warning(
                "reporter.confirm_terminal_status: status=%d",
                exc.response.status_code,
            )
        return _SendOutcome.DROPPED

    def _batch_failure_outcome(self, exc: Exception) -> _SendOutcome:
        """Map an ingest send failure to RETRY (transient) or DROPPED (terminal)."""
        return _SendOutcome.RETRY if _is_retryable_exc(exc) else _SendOutcome.DROPPED

    def _record_confirm_success(self) -> None:
        """Reset the confirm failure counter after a successful confirm POST."""
        self._consecutive_confirm_failures = 0

    def _record_confirm_failure(self, exc: Exception) -> None:
        """Track confirm send failures and escalate persistent outages."""
        self._consecutive_confirm_failures += 1
        count = self._consecutive_confirm_failures
        if count >= self._confirm_failure_threshold:
            _log_error(
                "reporter.confirm_send_persistent_failure: exc_type=%s consecutive_failures=%d",
                type(exc).__name__,
                count,
            )
        else:
            _log_warning(
                "reporter.confirm_send_failed: exc_type=%s",
                type(exc).__name__,
            )

    def _record_parseable_response(self) -> None:
        """Reset the unparseable counter after a successfully parsed ingest body."""
        self._consecutive_unparseable_responses = 0

    def _record_unparseable_response(self, exc: Exception) -> None:
        """Track unparseable ingest bodies and escalate persistent contract drift."""
        self._consecutive_unparseable_responses += 1
        count = self._consecutive_unparseable_responses
        if count >= self._unparseable_response_threshold:
            _log_error(
                "reporter.ingest_response_unparseable_persistent: "
                "exc_type=%s consecutive_failures=%d",
                type(exc).__name__,
                count,
            )
        else:
            _log_warning(
                "reporter.ingest_response_unparseable: exc_type=%s",
                type(exc).__name__,
            )

    def _parse_ingest_rejections(
        self, response: httpx.Response, batch_size: int
    ) -> _IngestRejections:
        """Parse and log a 202 body without publishing any disposition.

        The API returns 202 for every well-formed request; accepted events are
        already durable and rejected events are terminal — they reject
        identically on every resubmission until a pricing entry lands. Exact
        indexes let the drain dispose the selected owned events; a partial
        legacy index-less body preserves only its bounded loss count, while a
        full-batch legacy rejection proves every identity. Disposition is
        deliberately deferred to the drain so sync ownership release and
        publication can be one atomic operation with respect to close/seal.
        One WARNING per distinct (code, model) per batch keeps a fleet stuck
        on a single unpriced model from flooding logs. The server's message is
        logged verbatim (repr-escaped server-side) and never parsed.

        Fail-open: a malformed body must never raise into the flush loop —
        acknowledge the successful transport without guessing a disposition.
        """
        try:
            rejected = response.json()["rejected"]
            if not isinstance(rejected, list):
                # A falsy non-list ("rejected": null) must take the fail-open
                # path, not masquerade as a clean batch.
                raise TypeError(f"rejected is {type(rejected).__name__}, expected list")
            if not rejected:
                self._record_parseable_response()
                return _IngestRejections(_IngestRejectionKind.CLEAN)
            if len(rejected) > batch_size:
                # Contract violation — and a cap: a compromised/misrouted
                # server cannot flood logs or the groups dict with more
                # distinct (code, model) pairs than events were submitted.
                raise ValueError("server rejected more events than were submitted")
            # Aggregate before logging so a malformed entry can never leave a
            # half-logged flush behind. First message per group wins (the
            # server's message is keyed by the same (code, model) inputs).
            groups: dict[tuple[str, str], tuple[int, str]] = {}
            rejected_indexes: list[int] = []
            indexes_complete = True
            for rejection in rejected:
                key = (
                    _escape_control(str(rejection["code"])),
                    _escape_control(str(rejection["model"])),
                )
                count, message = groups.get(key, (0, _escape_control(str(rejection["message"]))))
                groups[key] = (count + 1, message)
                try:
                    raw_index = rejection["index"]
                    if type(raw_index) is not int:
                        raise TypeError("rejection index is not a JSON integer")
                    index = raw_index
                    if index < 0 or index >= batch_size:
                        raise ValueError("rejection index outside submitted batch")
                except KeyError:
                    # Compatibility with pre-index response fixtures/servers:
                    # retain the established count-only accounting when the
                    # exact rejected event cannot be identified safely.
                    indexes_complete = False
                else:
                    rejected_indexes.append(index)
            if indexes_complete and len(set(rejected_indexes)) != len(rejected_indexes):
                raise ValueError("duplicate rejection index")
            self._record_parseable_response()
        except Exception as exc:
            self._record_unparseable_response(exc)
            return _IngestRejections(_IngestRejectionKind.MALFORMED)
        result = (
            _IngestRejections(
                _IngestRejectionKind.EXACT,
                indexes=frozenset(rejected_indexes),
                count=len(rejected),
            )
            if indexes_complete
            else _IngestRejections(_IngestRejectionKind.LEGACY, count=len(rejected))
        )
        try:
            for (code, model), (count, message) in groups.items():
                _log_warning(
                    "reporter.ingest_events_rejected: code=%s model=%s count=%d message=%s",
                    code,
                    model,
                    count,
                    message,
                )
        except Exception:
            # The body parsed fine and the batch IS durable server-side — only
            # the host's logging stack raised (e.g. a user-installed Filter).
            # A fallback log would raise again; rejection detail is
            # best-effort, but letting this propagate would mislabel the batch
            # as a send failure in _send_batch's accounting.
            pass
        return result


class MetadataReporter(_ReporterBase):
    """Synchronous metadata reporter with a background daemon thread.

    Usage::

        reporter = MetadataReporter(api_url, api_key)
        reporter.report(event)  # non-blocking
        # ...
        reporter.close()  # flush remaining events

    Or as a context manager::

        with MetadataReporter(api_url, api_key) as reporter:
            reporter.report(event)
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 10_000,
        max_in_flight: int = 3,
        breaker_snapshots: Callable[[], list[tuple[ProviderName, CircuitBreakerState]]]
        | None = None,
        sdk_instance_id: str | None = None,
        breaker_reporting_enabled: bool = True,
        report_untracked_surfaces: bool = True,
        breaker_report_heartbeat: float = 60.0,
        control_plane_breaker: CircuitBreaker | None = None,
        max_send_attempts: int = 5,
        retry_backoff_base: float = 1.0,
        retry_backoff_cap: float = 60.0,
        shutdown_deadline: float = 5.0,
    ) -> None:
        super().__init__(
            api_url,
            api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            max_in_flight=max_in_flight,
            breaker_snapshots=breaker_snapshots,
            sdk_instance_id=sdk_instance_id,
            breaker_reporting_enabled=breaker_reporting_enabled,
            report_untracked_surfaces=report_untracked_surfaces,
            breaker_report_heartbeat=breaker_report_heartbeat,
            control_plane_breaker=control_plane_breaker,
            max_send_attempts=max_send_attempts,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_cap=retry_backoff_cap,
            shutdown_deadline=shutdown_deadline,
        )
        # ``_delivery_closed`` (base) is the early enqueue-refusal gate.
        # Completion is distinct and becomes true only at the END of seal's
        # ownership transaction. The close lock selects one final-drain sender
        # when worker-reentrant, external, and exit closes race.
        self._delivery_completed = False
        # Whole-close completion is later than spend sealing: lifecycle rescue
        # stays armed until the transport has also closed successfully.
        self._close_completed = False
        self._close_finalization_owner: threading.Thread | None = None
        self._transport_close_started = False
        self._transport_close_owner: threading.Thread | None = None
        # Every close request contributes one absolute deadline. Nested and
        # competing callers may only tighten this process-local minimum.
        self._close_state_lock = threading.Lock()
        self._close_state_changed = threading.Condition(self._close_state_lock)
        self._earliest_close_deadline: float | None = None
        self._close_lock = threading.RLock()
        self._worker_finalize_deadline: float | None = None
        self._http = httpx.Client(timeout=10.0)
        self._shutdown = threading.Event()
        self._in_flight_lock = threading.Lock()
        self._breaker_worker_lock = threading.Lock()
        self._untracked_worker_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        # Set by the fork handler so the next enqueue relaunches the (fork-killed)
        # flush thread. False otherwise, so a never-forked reporter never spawns
        # an unexpected thread from report().
        self._needs_thread_restart = False
        self._breaker_worker: threading.Thread | None = None
        self._untracked_worker: threading.Thread | None = None
        self._thread = self._launch_thread()
        # Exit flush: if the process exits without close(), the atexit hook runs
        # close() so queued spend is still delivered. The live flush thread keeps
        # this reporter alive (its bound-method target), so no finalizer is
        # needed on the sync path — close() is the whole story.
        register_sync_reporter(self)
        register_fork_reset(self)

    def _launch_thread(self) -> threading.Thread:
        """Create and start a fresh flush thread."""
        thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="solwyn-reporter",
        )
        thread.start()
        return thread

    def _ensure_thread(self) -> None:
        """Relaunch the flush thread if it is not running (e.g. after a fork).

        The flush thread starts in ``__init__``; a forked child inherits a DEAD
        thread. Restarting it lazily on the next enqueue — rather than inside the
        ``os.register_at_fork`` handler, where starting a thread is fragile —
        keeps delivery alive in the child. Guarded by ``_needs_thread_restart``
        (set only by the fork handler) so a never-forked reporter never spawns an
        unexpected thread; a no-op for a closed reporter.
        """
        if not self._needs_thread_restart or self._shutdown.is_set():
            return
        with self._thread_lock:
            if not self._needs_thread_restart or self._shutdown.is_set():
                return
            self._needs_thread_restart = False
            self._thread = self._launch_thread()

    def _reset_after_fork_in_child(self) -> None:
        """Repair a forked child: fresh locks + client; defer the thread relaunch.

        Threads do not survive ``fork()``, so the inherited flush thread is dead
        in the child. Starting a replacement INSIDE this fork handler is fragile
        (the child is in a delicate post-fork state), so the thread is instead
        relaunched lazily by ``_ensure_thread`` on the next enqueue. Locks
        possibly held by a now-absent thread are replaced, and the inherited
        client is abandoned (never closed — the parent owns those sockets).
        Queued spend items duplicated into the child by fork are deliberately
        KEPT: the server dedups them. Inherited advisory observations are
        discarded because parent and child mint different report IDs. A
        completed reporter stays terminal; an interrupted seal reopens only
        the internal delivery gate so the child's close/atexit owner can finish
        the retained queues (the public shutdown request remains set).
        """
        # Read the inherited flag pair without its old ownership lock: at-fork
        # child repair runs single-threaded, and that lock may be held forever
        # by a vanished parent thread. `_delivery_closed` is published at the
        # START of seal while `_delivery_completed` is published at its END.
        # The mismatched pair therefore means no surviving owner can finish the
        # inherited transition unless the child reopens the internal gate.
        interrupted_seal = self._delivery_closed and not self._delivery_completed
        self._breaker_project_lock = threading.Lock()
        self._breaker_report_lock = threading.Lock()
        if self._untracked_state is not None:
            self._untracked_state.reset_after_fork_in_child()
        self._receipt_fold_state.reset_after_fork_in_child()
        self._drop_lock = threading.Lock()
        self._in_flight_lock = threading.Lock()
        self._breaker_worker_lock = threading.Lock()
        self._untracked_worker_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._close_lock = threading.RLock()
        self._close_state_lock = threading.Lock()
        self._close_state_changed = threading.Condition(self._close_state_lock)
        self._ownership_lock = threading.Lock()
        if interrupted_seal:
            self._delivery_closed = False
        # The inherited client belongs to the parent, and any inherited close
        # deadline belonged to a vanished thread. The fresh child transport
        # gets a fresh completion attempt even when spend delivery was already
        # terminal in the parent snapshot.
        self._close_completed = False
        self._close_finalization_owner = None
        self._transport_close_started = False
        self._transport_close_owner = None
        self._earliest_close_deadline = None
        # In-hand items live on the parent's (now-absent) flush thread stack;
        # the parent delivers them. The child's view starts clean.
        self._in_hand = {}
        self._next_event_batch_token = 0
        self._event_batches_in_hand = {}
        self._next_settlement_token = 0
        self._settlements_in_hand = {}
        self._in_flight = 0
        self._final_delivery_started = False
        # The parent's worker cannot continue its close request in the child.
        # Leave incomplete delivery for the child's atexit rescue instead of
        # inheriting a deadline no live thread owns.
        self._worker_finalize_deadline = None
        self._breaker_worker = None
        self._untracked_worker = None
        self._http = httpx.Client(timeout=10.0)
        if not self._shutdown.is_set():
            # Replace the inherited Event and arm the lazy relaunch; a closed
            # reporter keeps its set Event and never relaunches.
            self._shutdown = threading.Event()
            self._needs_thread_restart = True

    def _notify_untracked_observation(self) -> None:
        """Wake the origin reporter without doing network I/O on the caller thread."""
        if self._untracked_state is None:
            return
        try:
            self._ensure_thread()
            if self._untracked_reports_due():
                self._start_untracked_cycle()
        except Exception:
            # Advisory activation must never affect the provider call.
            return

    def report(self, event: MetadataEvent) -> None:
        """Enqueue a metadata event for async reporting.  Non-blocking."""
        if self._shutdown.is_set():
            # Nothing will ever drain a post-close enqueue — count it like
            # report_confirm/report_settlement do instead of retaining it
            # silently (a late streaming on_complete can land here).
            self._fold_or_count_event_drop(event, "closed_enqueue")
            return
        self._ensure_thread()
        if not self._enqueue(event):
            self._fold_or_count_event_drop(event, "closed_enqueue")

    def _mark_delivery_completed_locked(self) -> None:
        """Publish sync seal completion at the end of its ownership transaction."""
        self._delivery_completed = True

    def _delivery_is_completed(self) -> bool:
        """Read sync completion through the same lock that publishes it."""
        with self._ownership_lock:
            return self._delivery_completed

    def _close_is_completed(self) -> bool:
        """Whether spend sealing and sync transport cleanup both finished."""
        with self._close_state_lock:
            return self._close_completed

    def _tighten_close_deadline(self, deadline: float) -> float:
        """Publish and return the earliest absolute sync close deadline."""
        with self._close_state_changed:
            current = self._earliest_close_deadline
            if current is None or deadline < current:
                current = deadline
                self._earliest_close_deadline = current
                self._close_state_changed.notify_all()
            return current

    def _effective_close_deadline(self, deadline: float | None) -> float | None:
        """Combine a call-local bound with every nested/competing close."""
        with self._close_state_lock:
            close_deadline = self._earliest_close_deadline
        if deadline is None:
            return close_deadline
        if close_deadline is None:
            return deadline
        return min(deadline, close_deadline)

    def _deadline_expired(self, deadline: float | None) -> bool:
        """Apply a newly tightened close deadline to already-running drains."""
        return super()._deadline_expired(self._effective_close_deadline(deadline))

    def _send_timeout(self, deadline: float | None) -> float:
        """Clamp sends to the latest view of the earliest close deadline."""
        return super()._send_timeout(self._effective_close_deadline(deadline))

    def _join_until_close_deadline(self, worker: threading.Thread, deadline: float | None) -> bool:
        """Join without self-deadlock and notice a competing tighter deadline."""
        if worker is threading.current_thread():
            return False
        while worker.is_alive():
            effective = self._effective_close_deadline(deadline)
            if effective is None:
                worker.join()
                continue
            remaining = effective - _monotonic()
            if remaining <= 0:
                return False
            # Thread.join cannot be interrupted when another caller tightens
            # the deadline, so poll briefly and re-read the shared minimum.
            worker.join(timeout=min(remaining, 0.01))
        return True

    def _acquire_close_until_deadline(self, deadline: float) -> bool:
        """Acquire the close RLock while observing a dynamically tighter bound."""
        while True:
            effective = self._effective_close_deadline(deadline)
            if effective is None:
                self._close_lock.acquire()
                return True
            remaining = effective - _monotonic()
            if remaining <= 0:
                return self._close_lock.acquire(blocking=False)
            if self._close_lock.acquire(timeout=min(remaining, 0.01)):
                return True

    def _close_transport(self) -> None:
        """Idempotently close the sync transport, leaving rescue armed on failure."""
        current = threading.current_thread()
        with self._close_state_changed:
            while self._transport_close_started and self._transport_close_owner is not current:
                deadline = self._earliest_close_deadline
                if deadline is None:
                    self._close_state_changed.wait()
                    continue
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    return
                self._close_state_changed.wait(timeout=min(remaining, 0.01))
            if self._close_completed or self._transport_close_owner is current:
                return
            self._transport_close_started = True
            self._transport_close_owner = current
        closed = False
        try:
            self._http.close()
            closed = True
        except Exception as exc:
            _log_warning("reporter.transport_close_failed: exc_type=%s", type(exc).__name__)
        finally:
            with self._close_state_changed:
                self._transport_close_started = False
                self._transport_close_owner = None
                if closed:
                    self._close_completed = True
                self._close_state_changed.notify_all()

    def _claim_close_finalization(self, deadline: float, *, wait: bool) -> _CloseFinalizationClaim:
        """Own spend finalization through advisory work and transport close."""
        current = threading.current_thread()
        with self._close_state_changed:
            while True:
                if self._close_completed:
                    return _CloseFinalizationClaim.COMPLETED
                owner = self._close_finalization_owner
                if owner is None:
                    self._close_finalization_owner = current
                    return _CloseFinalizationClaim.ACQUIRED
                if owner is current:
                    return _CloseFinalizationClaim.REENTRANT
                if not wait:
                    return _CloseFinalizationClaim.BUSY
                close_deadline = self._earliest_close_deadline
                remaining = (
                    deadline if close_deadline is None else min(deadline, close_deadline)
                ) - _monotonic()
                if remaining <= 0:
                    return _CloseFinalizationClaim.EXPIRED
                self._close_state_changed.wait(timeout=min(remaining, 0.01))

    def _release_close_finalization(self) -> None:
        """Release whole-close ownership after completion or a retryable failure."""
        current = threading.current_thread()
        with self._close_state_changed:
            if self._close_finalization_owner is current:
                self._close_finalization_owner = None
                self._close_state_changed.notify_all()

    def _request_worker_finalize(self, deadline: float) -> None:
        """Ask the flush worker to finalize after its current owned batch."""
        with self._close_state_lock:
            if self._close_completed:
                return
            current = self._worker_finalize_deadline
            self._worker_finalize_deadline = deadline if current is None else min(current, deadline)

    def _take_worker_finalize_deadline(self) -> float | None:
        with self._close_state_lock:
            deadline = self._worker_finalize_deadline
            self._worker_finalize_deadline = None
            return deadline

    def _finish_worker_requested_close(self) -> None:
        """Let whichever reporter-owned worker exits first finish close."""
        deadline = self._take_worker_finalize_deadline()
        if deadline is None:
            return
        try:
            # If an external closer already owns finalization it may be joining
            # this worker. Waiting here would create a join/ownership cycle;
            # that owner already carries the same globally-tightened deadline.
            self._finish_close(deadline, wait_for_owner=False)
        except Exception as exc:
            # _finish_close seals spend before propagating. Worker teardown and
            # arbitrary host logging callbacks remain fail-soft.
            _log_warning(
                "reporter.worker_finalization_failed: exc_type=%s",
                type(exc).__name__,
            )

    def close(self, timeout: float | None = None) -> None:
        """Flush remaining events and shut down within a single deadline.

        ``timeout`` (default ``self.shutdown_deadline``) bounds the WHOLE
        shutdown chain — thread join, final flush, and breaker report cycle all
        share one monotonic ``deadline``. Work still queued when it is reached is
        counted ``shutdown_deadline`` and dropped rather than paying a serial
        per-request timeout chain against a black-holed control plane.
        """
        budget = self.shutdown_deadline if timeout is None else timeout
        deadline = self._tighten_close_deadline(_monotonic() + budget)
        # Serialize the stop request with cadence-triggered breaker launches.
        # The finalization owner snapshots advisory workers after ingest stops.
        with self._breaker_worker_lock:
            self._shutdown.set()

        # A host logging handler can reenter close() from this reporter's own
        # flush thread. Joining ourselves is forbidden, and continuing into a
        # recursive final drain while the current batch is still in hand would
        # make close's ownership topology needlessly reentrant. The worker keeps
        # the reporter alive, finishes that disposition, then runs the shared
        # finalizer itself before its bound-method reference can disappear.
        current = threading.current_thread()
        if current in (self._thread, self._breaker_worker, self._untracked_worker):
            self._request_worker_finalize(deadline)
            return

        # Join the ingest loop within the remaining budget. If the join times
        # out, the final flush below STILL runs: deque ops are thread-safe and a
        # duplicate send is server-deduped via the idempotency ledger.
        self._join_until_close_deadline(self._thread, deadline)
        self._finish_close(deadline)

    def _finish_close(self, deadline: float, *, wait_for_owner: bool = True) -> None:
        """Run the one-owner final drain, or force-seal at a raced deadline.

        Whole-close ownership selects the ONLY final sender through spend seal,
        advisory cycles, and transport cleanup. A competing caller contributes
        its tighter deadline and waits only within that bound; it never closes
        the transport underneath the owner. Drop callbacks run after spend
        ownership locks are released.
        """
        deadline = self._tighten_close_deadline(deadline)
        claim = self._claim_close_finalization(deadline, wait=wait_for_owner)
        if claim is _CloseFinalizationClaim.EXPIRED:
            # The whole-close owner alone may run advisory work or close the
            # transport. The expired waiter still atomically terminalizes spend
            # so a paused daemon owner can never be the sole remaining record.
            self._seal_delivery()
            return
        if claim is not _CloseFinalizationClaim.ACQUIRED:
            return
        try:
            self._finish_close_owned(deadline)
        finally:
            self._release_close_finalization()

    def _finish_close_owned(self, deadline: float) -> None:
        """Complete the final drain while holding whole-close ownership."""
        acquired = self._acquire_close_until_deadline(deadline)
        if not acquired:
            self._seal_delivery()
            self._close_transport()
            return

        already_completed = False
        sealed_here = False
        close_error: Exception | None = None
        active_breaker_worker: threading.Thread | None = None
        try:
            deadline = self._effective_close_deadline(deadline) or deadline
            already_completed = self._delivery_is_completed()
            if not already_completed:
                with self._breaker_worker_lock:
                    active_breaker_worker = self._breaker_worker
                    if active_breaker_worker is not None and not active_breaker_worker.is_alive():
                        active_breaker_worker = None
                try:
                    # Terminal fold ownership precedes the final worker. Even a
                    # zero deadline leaves no aggregate without a disposition.
                    self._drain_receipt_folds_to_queue(final=True)
                    if not self._delivery_is_completed():
                        self._final_flush_bounded(
                            self._effective_close_deadline(deadline) or deadline
                        )
                except Exception as exc:
                    close_error = exc
                # Success, exception, or deadline race all end in the same
                # ownership mutation before completion becomes observable.
                sealed_here = self._seal_delivery(emit=False)
        finally:
            self._close_lock.release()

        if sealed_here:
            self._maybe_log_drops(force=True)
        if already_completed or not sealed_here:
            self._close_transport()
            return
        if close_error is not None:
            self._close_transport()
            raise close_error
        try:
            with self._untracked_worker_lock:
                active_untracked_worker = self._untracked_worker
                if active_untracked_worker is not None and not active_untracked_worker.is_alive():
                    active_untracked_worker = None
            if active_untracked_worker is not None:
                self._join_until_close_deadline(active_untracked_worker, deadline)
                # The worker clears its coalescing slot before it exits. Re-read
                # that slot after the join: the pre-join local must not suppress
                # one final cycle for observations that missed its snapshot.
                with self._untracked_worker_lock:
                    active_untracked_worker = self._untracked_worker
                    if (
                        active_untracked_worker is not None
                        and not active_untracked_worker.is_alive()
                    ):
                        active_untracked_worker = None
            if (
                active_untracked_worker is None
                and not self._deadline_expired(deadline)
                and self._untracked_reports_due()
            ):
                final_untracked_worker = self._start_untracked_cycle(
                    during_shutdown=True,
                    deadline=self._effective_close_deadline(deadline),
                )
                if final_untracked_worker is not None:
                    self._join_until_close_deadline(final_untracked_worker, deadline)

            if active_breaker_worker is not None:
                self._join_until_close_deadline(active_breaker_worker, deadline)
            if active_breaker_worker is None or not self._deadline_expired(deadline):
                final_breaker_worker = self._start_breaker_cycle(
                    during_shutdown=True,
                    deadline=self._effective_close_deadline(deadline),
                    force=True,
                )
                if final_breaker_worker is not None:
                    self._join_until_close_deadline(final_breaker_worker, deadline)
        finally:
            self._close_transport()

    def _final_flush_bounded(self, deadline: float) -> None:
        """Run close()'s final drain on a worker joined at ``deadline``.

        httpx timeouts bound individual socket operations, not total response
        time, so a slow-drip response could hold an INLINE final flush past the
        deadline indefinitely. The worker is a daemon: if it is still stuck at
        the deadline, close() proceeds — ``_seal_delivery`` claims and counts
        the worker's in-hand items, and ``_http.close()`` tears down the stuck
        transport under it. At interpreter shutdown new threads can be refused;
        fall back to the inline flush there (per-request clamps are then the
        only bound).
        """

        def _run() -> None:
            try:
                self._flush_remaining(deadline=deadline, final=True)
            except Exception as exc:
                # A raise on the worker would otherwise vanish into the thread
                # excepthook; the seal still accounts whatever it left behind.
                _log_warning("reporter.final_flush_failed: exc_type=%s", type(exc).__name__)

        try:
            worker = threading.Thread(target=_run, daemon=True, name="solwyn-final-flush")
            worker.start()
        except RuntimeError:
            _run()
            return
        self._join_until_close_deadline(worker, deadline)

    def __enter__(self) -> MetadataReporter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _flush_loop(self) -> None:
        """Background thread: periodically flush batches to the cloud."""
        try:
            while not self._shutdown.is_set():
                if self._shutdown.wait(timeout=self.flush_interval):
                    break
                try:
                    self._flush_remaining()
                except Exception as exc:
                    # The flush loop must survive anything a drain raises: a
                    # dead worker strands ALL queued spend until overflow.
                    _log_warning("reporter.flush_cycle_failed: exc_type=%s", type(exc).__name__)
                if self._breaker_reports_due():
                    self._start_breaker_cycle()
                if self._untracked_reports_due():
                    self._start_untracked_cycle()
        finally:
            self._finish_worker_requested_close()

    def _flush_remaining(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Flush queued confirms, settlements, then metadata events in batches.

        ``final`` is the shutdown/exit mode: backoff gates are ignored, a RETRY
        outcome drops (there is no later cycle), and a HELD outcome drops the
        remainder (never hammer a known-down control plane while exiting).
        ``deadline`` (a monotonic instant) caps the whole chain; work still
        queued when it is reached is counted ``shutdown_deadline`` and dropped.
        """
        # A prior successful metadata send is the recovery proof. Consume that
        # one-shot gate before ordinary drains; final close gets exactly one
        # unconditional attempt inside the same deadline.
        self._drain_receipt_folds_to_queue(final=final)
        self._drain_confirms(deadline=deadline, final=final)
        self._drain_settlements(deadline=deadline, final=final)
        if self._drain_event_batches(deadline=deadline, final=final):
            self._receipt_fold_state.note_cycle_success()
        self._maybe_log_drops(force=final)

    # ------------------------------------------------------------------
    # Shutdown ownership: in-hand drain items
    # ------------------------------------------------------------------

    def _pop_in_hand(self, queue: collections.deque[_T], kind: str) -> _T | None:
        """Atomically pop the head AND mark it in hand; None once sealed/empty.

        Pop and ownership registration must be one step under the ownership
        lock: with a gap between them, ``_seal_delivery`` could run in the gap
        and see the popped item in neither the queue nor ``_in_hand`` — close()
        would return with that spend unaccounted (#5 review pin). Once sealed
        the pop is refused outright: the seal already claimed and counted the
        queue's contents.
        """
        with self._ownership_lock:
            if self._delivery_closed:
                return None
            try:
                item = queue.popleft()
            except IndexError:
                return None
            self._in_hand[kind] = self._in_hand.get(kind, 0) + 1
            return item

    def _pop_settlement_in_hand(self) -> _ClaimedSettlement | None:
        """Atomically pop and token-register one exact settlement pair."""
        with self._ownership_lock:
            if self._delivery_closed:
                return None
            try:
                pending = self._settlement_queue.popleft()
            except IndexError:
                return None
            self._next_settlement_token += 1
            token = _SettlementToken(self._next_settlement_token)
            self._settlements_in_hand[token] = pending
            return _ClaimedSettlement(token, pending)

    def _resolve_settlement_to_event_queue(
        self,
        claim: _ClaimedSettlement,
        *,
        confirm_drop_reason: str | None = None,
    ) -> bool:
        """Atomically transfer one exact pair from token to event ownership."""
        resolved = False
        with self._ownership_lock:
            pending = self._settlements_in_hand.pop(claim.token, None)
            if pending is not None:
                if confirm_drop_reason is not None:
                    self._record_drop("settlement_confirm", confirm_drop_reason)
                self._enqueue_event_locked(pending.event)
                resolved = True
        self._maybe_log_drops()
        return resolved

    def _resolve_settlement_bulk_to_event_queue(
        self, claim: _ClaimedSettlement, *, confirm_drop_reason: str
    ) -> bool:
        """Atomically transfer one token and every queued pair to events."""
        resolved = False
        with self._ownership_lock:
            pending = self._settlements_in_hand.pop(claim.token, None)
            if pending is not None:
                settlements = [pending, *self._settlement_queue]
                self._settlement_queue.clear()
                self._record_drop("settlement_confirm", confirm_drop_reason, n=len(settlements))
                for item in settlements:
                    self._enqueue_event_locked(item.event)
                resolved = True
        self._maybe_log_drops()
        return resolved

    def _dispose_queued_settlements(self, reason: str) -> None:
        """Atomically terminally dispose every pair still in the queue."""
        with self._ownership_lock:
            settlements = list(self._settlement_queue)
            self._settlement_queue.clear()
            self._dispose_settlements(settlements, reason, emit=False)
        self._maybe_log_drops()

    def _dispose_queued_events(self, reason: str) -> None:
        """Atomically claim and terminally dispose every queued event.

        A final drain may run concurrently with close's seal after its bounded
        join expires. Queue removal and the fold/drop mutation must therefore
        be one ownership transaction: a local list alone is not an owner the
        seal can observe, and the daemon worker is not guaranteed to resume.
        Warning emission remains outside the ownership lock.
        """
        with self._ownership_lock:
            events = list(self._queue)
            self._queue.clear()
            self._dispose_pending_events(events, reason, emit=False)
        self._maybe_log_drops()

    def _pop_batch_in_hand(self, *, now: float, final: bool) -> _ClaimedEventBatch | None:
        """Atomically pop and token-register one due event-batch prefix.

        Same seal-gap rationale as ``_pop_in_hand``, for the batched event
        drain. A distinct token owns each prefix because concurrent background
        and close/final drains can complete out of order. The inner IndexError
        guards are defensive against an unexpectedly empty deque; every sync
        production removal is serialized by this same ownership lock.
        """
        with self._ownership_lock:
            if self._delivery_closed:
                return None
            prefix: list[_PendingEvent] = []
            while len(prefix) < self.batch_size:
                try:
                    head = self._queue[0]
                except IndexError:
                    break
                if not final and head.next_attempt_at > now:
                    break
                try:
                    prefix.append(self._queue.popleft())
                except IndexError:
                    break
            if not prefix:
                return None
            self._next_event_batch_token += 1
            token = _EventBatchToken(self._next_event_batch_token)
            pending = tuple(prefix)
            self._event_batches_in_hand[token] = pending
            return _ClaimedEventBatch(token, pending)

    def _resolve_confirm(
        self,
        *,
        drop_reason: str | None = None,
        requeue: Callable[[], None] | None = None,
    ) -> None:
        """Release one in-hand confirm with its terminal disposition.

        Confirm identities are fungible, so a count is sufficient. Loss
        accounting nevertheless belongs in the SAME ownership transaction as
        release: close may seal after its bounded worker join and the daemon is
        not guaranteed to resume. ``requeue`` runs only while delivery is open;
        if seal already claimed the confirm, ``owned`` is zero and this is a
        no-op. Warning emission happens after ownership is released.
        """
        recorded = False
        with self._ownership_lock:
            held = self._in_hand.get("confirm", 0)
            owned = min(held, 1)
            if owned:
                self._in_hand["confirm"] = held - owned
            if requeue is not None and not self._delivery_closed:
                requeue()
                return
            if owned and drop_reason is not None:
                self._record_drop("confirm", drop_reason, n=owned)
                recorded = True
        if recorded:
            self._maybe_log_drops()

    def _dispose_queued_confirms(self, reason: str) -> None:
        """Atomically claim and count every standalone queued confirm."""
        with self._ownership_lock:
            count = _drain_count(self._confirm_queue)
            self._record_drop("confirm", reason, n=count)
        self._maybe_log_drops()

    def _resolve_confirm_and_queue(self, reason: str) -> None:
        """Atomically dispose the in-hand confirm and its queued FIFO tail."""
        with self._ownership_lock:
            held = self._in_hand.get("confirm", 0)
            owned = min(held, 1)
            if owned:
                self._in_hand["confirm"] = held - owned
            count = owned + _drain_count(self._confirm_queue)
            self._record_drop("confirm", reason, n=count)
        self._maybe_log_drops()

    def _resolve_sent_event_batch(
        self, claim: _ClaimedEventBatch, result: _BatchSendResult
    ) -> bool:
        """Atomically release a sent sync batch and publish its rejections.

        If close/seal already claimed the batch, ``owned`` is zero and the
        response publishes nothing. A recovery proof requires an actual clean
        response and full ownership of the sent prefix.
        """
        clean = False
        with self._ownership_lock:
            owned = self._event_batches_in_hand.pop(claim.token, None)
            if owned is not None:
                self._publish_ingest_rejections(
                    [pending.event for pending in owned], result.rejections, emit=False
                )
                clean = result.rejections.kind is _IngestRejectionKind.CLEAN
        self._maybe_log_drops()
        return clean

    def _resolve_retry_event_batch(
        self,
        claim: _ClaimedEventBatch,
        *,
        keep: list[_PendingEvent],
    ) -> bool:
        """Resolve a retry result by token, atomically requeueing its survivors."""
        resolved = False
        with self._ownership_lock:
            owned = self._event_batches_in_hand.pop(claim.token, None)
            if owned is not None:
                keep_ids = {id(pending) for pending in keep}
                owned_keep = [pending for pending in owned if id(pending) in keep_ids]
                owned_dropped = [pending for pending in owned if id(pending) not in keep_ids]
                if not self._delivery_closed:
                    self._queue.extendleft(reversed(owned_keep))
                    self._dispose_pending_events(owned_dropped, "retry_exhausted", emit=False)
                    resolved = True
                else:
                    # Defensive invariant: seal normally clears every token
                    # while it marks delivery closed.
                    self._dispose_pending_events(list(owned), "shutdown_deadline", emit=False)
        self._maybe_log_drops()
        return resolved

    def _resolve_dropped_event_batch(self, claim: _ClaimedEventBatch, reason: str) -> bool:
        """Resolve one terminal batch by token and publish each owned identity."""
        resolved = False
        with self._ownership_lock:
            owned = self._event_batches_in_hand.pop(claim.token, None)
            if owned is not None:
                self._dispose_pending_events(list(owned), reason, emit=False)
                resolved = True
        self._maybe_log_drops()
        return resolved

    def _park_confirm(self, pending: _PendingConfirm) -> None:
        """Requeue a still-retryable confirm head — or count it once sealed."""
        self._resolve_confirm(
            drop_reason="shutdown_deadline",
            requeue=lambda: self._confirm_queue.appendleft(pending),
        )

    def _park_settlement(self, claim: _ClaimedSettlement) -> None:
        """Requeue a still-retryable settlement head — or count the pair."""
        with self._ownership_lock:
            pending = self._settlements_in_hand.pop(claim.token, None)
            if pending is None:
                return
            if not self._delivery_closed:
                self._settlement_queue.appendleft(pending)
                return
            self._dispose_settlements([pending], "shutdown_deadline", emit=False)
        self._maybe_log_drops()

    # ------------------------------------------------------------------
    # Drains
    # ------------------------------------------------------------------

    def _drain_confirms(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Send due confirms in FIFO order; a retrying head parks the queue."""
        while self._confirm_queue:
            if self._deadline_expired(deadline):
                self._dispose_queued_confirms("shutdown_deadline")
                break
            # Atomic pop-and-claim, never peek-then-pop: once close()'s bounded
            # join times out, close/atexit drains CONCURRENTLY with the flush
            # thread — two drainers peeking one head would double-send it. The
            # ownership claim rides the same lock so _seal_delivery can never
            # observe the item in neither the queue nor _in_hand.
            pending = self._pop_in_hand(self._confirm_queue, "confirm")
            if pending is None:
                break
            if not final and pending.next_attempt_at > _monotonic():
                self._park_confirm(pending)
                break  # head still backing off — nothing behind is due earlier
            outcome = self._send_confirm(pending.request, timeout=self._send_timeout(deadline))
            if outcome is _SendOutcome.SENT:
                self._resolve_confirm()
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    self._resolve_confirm_and_queue("exit_breaker_open")
                else:
                    self._park_confirm(pending)
                break
            if outcome is _SendOutcome.RETRY:
                pending.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.attempts)
                if finished or final:
                    self._resolve_confirm(drop_reason="retry_exhausted")
                    continue
                pending.next_attempt_at = next_at
                self._park_confirm(pending)
                break  # FIFO: nothing behind a backing-off head may jump it
            self._resolve_confirm(drop_reason="terminal_status")

    def _drain_settlements(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Resolve each settlement's confirm first, then hand its event to the
        event queue (confirm-before-metadata order per item is load-bearing).
        A retrying head parks the queue behind it (FIFO)."""
        while self._settlement_queue:
            if self._deadline_expired(deadline):
                self._dispose_queued_settlements("shutdown_deadline")
                break
            # Atomic pop-and-claim — see _drain_confirms for the rationale.
            claim = self._pop_settlement_in_hand()
            if claim is None:
                break
            pending = claim.pending
            if not final and pending.confirm.next_attempt_at > _monotonic():
                self._park_settlement(claim)
                break
            outcome = self._send_confirm(
                pending.confirm.request, timeout=self._send_timeout(deadline)
            )
            if outcome is _SendOutcome.SENT:
                self._resolve_settlement_to_event_queue(claim)
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    # The confirms are undeliverable (control plane known-down),
                    # but ingest is NOT breaker-gated: hand every event to the
                    # event drain so the durable spend truth still gets its
                    # deadline-bounded exit attempt.
                    self._resolve_settlement_bulk_to_event_queue(
                        claim, confirm_drop_reason="exit_breaker_open"
                    )
                else:
                    self._park_settlement(claim)
                break
            if outcome is _SendOutcome.RETRY:
                pending.confirm.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.confirm.attempts)
                if finished or final:
                    self._resolve_settlement_to_event_queue(
                        claim, confirm_drop_reason="retry_exhausted"
                    )
                    continue
                pending.confirm.next_attempt_at = next_at
                self._park_settlement(claim)
                break
            # Terminal confirm: the event is still the durable spend truth.
            self._resolve_settlement_to_event_queue(claim, confirm_drop_reason="terminal_status")

    def _drain_event_batches(self, *, deadline: float | None = None, final: bool = False) -> bool:
        """Send due metadata events in batches; requeue a failed batch to front."""
        sent_clean_batch = False
        cycle_clean = True
        while self._queue:
            if self._deadline_expired(deadline):
                cycle_clean = False
                self._dispose_queued_events("shutdown_deadline")
                break
            with self._in_flight_lock:
                if self._in_flight >= self.max_in_flight:
                    break
            # Atomic pop-and-claim of the whole due prefix — see
            # _drain_confirms for the seal-gap rationale.
            claim = self._pop_batch_in_hand(now=_monotonic(), final=final)
            if claim is None:
                break
            prefix = list(claim.pending)
            result = self._send_batch([p.event for p in prefix], deadline=deadline)
            if result.outcome is _SendOutcome.SENT:
                clean = self._resolve_sent_event_batch(claim, result)
                sent_clean_batch = clean or sent_clean_batch
                cycle_clean = clean and cycle_clean
                continue
            if result.outcome is _SendOutcome.RETRY:
                cycle_clean = False
                keep: list[_PendingEvent] = []
                for p in prefix:
                    p.attempts += 1
                    finished, next_at = self._resolve_retryable(attempts=p.attempts)
                    if not (finished or final):
                        p.next_attempt_at = next_at
                        keep.append(p)
                self._resolve_retry_event_batch(claim, keep=keep)
                break
            cycle_clean = False
            self._resolve_dropped_event_batch(claim, "terminal_status")
        return sent_clean_batch and cycle_clean

    def _start_breaker_cycle(
        self,
        *,
        during_shutdown: bool = False,
        deadline: float | None = None,
        force: bool = False,
    ) -> threading.Thread | None:
        """Start one tracked breaker cycle, or coalesce into the active one."""
        with self._breaker_worker_lock:
            if self._shutdown.is_set() and not during_shutdown:
                return None
            worker = self._breaker_worker
            if worker is not None and worker.is_alive():
                return worker

            def _run() -> None:
                try:
                    self._flush_breaker_reports(deadline=deadline, force=force)
                finally:
                    with self._breaker_worker_lock:
                        if self._breaker_worker is threading.current_thread():
                            self._breaker_worker = None
                    self._finish_worker_requested_close()

            worker = threading.Thread(
                target=_run,
                daemon=True,
                name="solwyn-breaker-reporter",
            )
            self._breaker_worker = worker
            worker.start()
            return worker

    def _flush_breaker_reports(self, deadline: float | None = None, *, force: bool = False) -> None:
        """POST current breaker snapshots independently and drop every failure.

        Bounded by the shared shutdown ``deadline`` when set: remaining providers
        are skipped once it is reached. Breaker snapshots are advisory — dropping
        them is fine and is NOT counted as a spend drop.
        """
        for project_id, report in self._build_breaker_reports(force=force):
            if self._deadline_expired(deadline):
                return
            try:
                response = self._http.post(
                    f"{self.api_url}/api/v1/projects/{project_id}/providers/breaker-reports",
                    json=report.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=self._send_timeout(deadline),
                )
                response.raise_for_status()
                with self._breaker_report_lock:
                    self._breaker_last_sent[report.provider.value] = (
                        report.state,
                        report.failure_count,
                        report.success_count,
                    )
            except Exception as exc:
                if handle_read_only_key_error(exc):
                    # A read-only key denies every write: end the cycle instead
                    # of posting the remaining doomed snapshots.
                    return
                _log_warning(
                    "reporter.breaker_send_failed: provider=%s exc_type=%s",
                    report.provider.value,
                    type(exc).__name__,
                )

    def _start_untracked_cycle(
        self,
        *,
        during_shutdown: bool = False,
        deadline: float | None = None,
    ) -> threading.Thread | None:
        """Start one advisory report cycle, coalescing concurrent flush ticks."""
        if self._untracked_state is None:
            return None
        with self._untracked_worker_lock:
            if (self._shutdown.is_set() and not during_shutdown) or self._deadline_expired(
                deadline
            ):
                return None
            worker = self._untracked_worker
            if worker is not None and worker.is_alive():
                return worker

            def _run() -> None:
                try:
                    self._flush_untracked_reports(deadline=deadline)
                finally:
                    with self._untracked_worker_lock:
                        if self._untracked_worker is threading.current_thread():
                            self._untracked_worker = None
                    # A new key can arrive after this cycle took its snapshot.
                    # Schedule its cycle once this worker no longer occupies
                    # the coalescing slot.
                    if not self._shutdown.is_set() and self._untracked_reports_due():
                        self._start_untracked_cycle()
                    self._finish_worker_requested_close()

            worker = threading.Thread(
                target=_run,
                daemon=True,
                name="solwyn-untracked-surface-reporter",
            )
            self._untracked_worker = worker
            worker.start()
            return worker

    def _flush_untracked_reports(self, *, deadline: float | None = None) -> None:
        """POST eligible deltas in advisory batches and silently drop failures."""
        built_reports = self._build_untracked_reports()
        for offset in range(0, len(built_reports), _UNTRACKED_REPORT_BATCH_SIZE):
            if self._deadline_expired(deadline):
                return
            batch = built_reports[offset : offset + _UNTRACKED_REPORT_BATCH_SIZE]
            if not batch:
                continue
            self._mark_untracked_reports_attempted(batch)
            try:
                effective_deadline = self._effective_close_deadline(deadline)
                response = self._http.post(
                    f"{self.api_url}/api/v1/untracked-surfaces",
                    json=[built.report.model_dump(mode="json") for built in batch],
                    headers=self._auth_headers(),
                    timeout=(
                        10.0
                        if effective_deadline is None
                        else super()._send_timeout(effective_deadline)
                    ),
                )
                response.raise_for_status()
            except Exception:
                continue
            self._mark_untracked_reports_sent(batch)

    def _send_confirm(
        self, confirm_request: BudgetConfirmRequest, *, timeout: float = 5.0
    ) -> _SendOutcome:
        """Send one confirm request and return its delivery outcome.

        Breaker-refused admission returns HELD (the caller keeps the item for a
        later cycle); a transient failure returns RETRY; a terminal status or
        poison item returns DROPPED. Read-only-key is a policy skip, not a loss,
        so it returns SENT.
        """
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            # Control plane known-down: hold this confirm for a later cycle
            # without paying the timeout. Does not touch the confirm-failure
            # counter or the breaker.
            _log_debug("reporter.confirm_held_breaker_open")
            return _SendOutcome.HELD
        try:
            resp = self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=confirm_request.model_dump(mode="json"),
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            if breaker is not None:
                breaker.record_success()
            self._record_confirm_success()
            return _SendOutcome.SENT
        except Exception as exc:
            if handle_read_only_key_error(exc):
                if breaker is not None:
                    breaker.record_success()
                return _SendOutcome.SENT
            if breaker is not None:
                breaker.record_failure()
            self._record_confirm_failure(exc)
            return self._confirm_failure_outcome(exc)
        finally:
            # Cancellation (or any BaseException) bypasses the handler above; a
            # consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a success/failure verdict
            # has already released the slot.
            if breaker is not None:
                breaker.release_probe(admission)

    def _send_batch(
        self, batch: list[MetadataEvent], *, deadline: float | None = None
    ) -> _BatchSendResult:
        """Send a batch and return transport plus parsed member dispositions.

        Ingest is deliberately NOT control-plane-breaker-guarded: opening the
        enforcement breaker (which flips budget checks to their fail-open
        posture) on an ingest blip would be a worse failure mode than a delayed
        batch. Ingest self-paces via the retry backoff instead.

        ``deadline`` clamps the request into the shutdown window; without it a
        black-holed control plane made close() overrun its budget by the full
        10s client timeout (P0 review finding).
        """
        effective_deadline = self._effective_close_deadline(deadline)
        timeout = 10.0 if effective_deadline is None else super()._send_timeout(effective_deadline)
        with self._in_flight_lock:
            self._in_flight += 1
        try:
            payload = [e.model_dump(mode="json") for e in batch]
            resp = self._http.post(
                f"{self.api_url}/api/v1/metadata/ingest",
                json=payload,
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            rejections = self._parse_ingest_rejections(resp, len(batch))
            return _BatchSendResult(_SendOutcome.SENT, rejections)
        except Exception as exc:
            if handle_read_only_key_error(exc):
                return _BatchSendResult(
                    _SendOutcome.SENT, _IngestRejections(_IngestRejectionKind.MALFORMED)
                )
            # Log only the exception's class name — the type-name-only
            # convention every other except-block follows. Safe here even though
            # the batch is content-free; keeps the privacy contract uniform.
            _log_warning(
                "Failed to send metadata batch (%d events): %s",
                len(batch),
                type(exc).__name__,
            )
            return _BatchSendResult(
                self._batch_failure_outcome(exc),
                _IngestRejections(_IngestRejectionKind.MALFORMED),
            )
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1

    def report_confirm(self, request: BudgetConfirmRequest) -> None:
        """Fire-and-forget a confirm request onto the flush queue.

        Called from stream completion callbacks so the user's thread
        never blocks on Solwyn HTTP. The flush loop picks up confirm
        requests alongside metadata events.
        """
        if self._shutdown.is_set():
            self._count_drop("confirm", "closed_enqueue")
            return
        self._ensure_thread()
        enqueued = self._enqueue_owned(
            self._confirm_queue, _PendingConfirm(request), _MAX_PENDING_CONTROL, "confirm"
        )
        if not enqueued:
            self._count_drop("confirm", "closed_enqueue")

    def report_settlement(self, request: BudgetConfirmRequest, event: MetadataEvent) -> None:
        """Fire-and-forget a stream settlement as one ordered queue item."""
        if self._shutdown.is_set():
            self._count_drop("settlement_confirm", "closed_enqueue")
            self._fold_or_count_event_drop(event, "closed_enqueue")
            return
        self._ensure_thread()
        enqueued = self._enqueue_settlement_owned(
            _PendingSettlement(_PendingConfirm(request), event)
        )
        if not enqueued:
            self._count_drop("settlement_confirm", "closed_enqueue")
            self._fold_or_count_event_drop(event, "closed_enqueue")


class AsyncMetadataReporter(_ReporterBase):
    """Asynchronous metadata reporter using asyncio.create_task.

    Usage::

        async with AsyncMetadataReporter(api_url, api_key) as reporter:
            reporter.report(event)
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 10_000,
        max_in_flight: int = 3,
        breaker_snapshots: Callable[[], list[tuple[ProviderName, CircuitBreakerState]]]
        | None = None,
        sdk_instance_id: str | None = None,
        breaker_reporting_enabled: bool = True,
        report_untracked_surfaces: bool = True,
        breaker_report_heartbeat: float = 60.0,
        control_plane_breaker: CircuitBreaker | None = None,
        max_send_attempts: int = 5,
        retry_backoff_base: float = 1.0,
        retry_backoff_cap: float = 60.0,
        shutdown_deadline: float = 5.0,
    ) -> None:
        super().__init__(
            api_url,
            api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            max_in_flight=max_in_flight,
            breaker_snapshots=breaker_snapshots,
            sdk_instance_id=sdk_instance_id,
            breaker_reporting_enabled=breaker_reporting_enabled,
            report_untracked_surfaces=report_untracked_surfaces,
            breaker_report_heartbeat=breaker_report_heartbeat,
            control_plane_breaker=control_plane_breaker,
            max_send_attempts=max_send_attempts,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_cap=retry_backoff_cap,
            shutdown_deadline=shutdown_deadline,
        )
        self._http = httpx.AsyncClient(timeout=10.0)
        self._shutdown_event: asyncio.Event | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._breaker_task: asyncio.Task[None] | None = None
        self._untracked_task: asyncio.Task[None] | None = None
        # Set by close(); once closed, enqueues are dropped and start() fails
        # loud. Distinct from _shutdown_event, which only exists once a flush
        # loop has run — a never-started reporter has no shutdown event but can
        # still be closed.
        self._closed = False
        # Set only when close() FINISHES its flush chain. _closed alone must
        # not disarm the lifecycle rescue paths: a close() cancelled at its
        # first await has flushed nothing, so the atexit hook keys on this.
        self._close_completed = False
        # Latches the one-per-instance "enqueued with no running loop" warning
        # so a caller that never enters an event loop is warned once, not per
        # event.
        self._warned_no_loop = False
        # Set by register_async_reporter: a GC finalizer covering the case where
        # this reporter is dropped before its flush loop ever ran. close()
        # detaches it so nothing double-flushes.
        self._finalizer: weakref.finalize[..., AsyncMetadataReporter] | None = None
        register_async_reporter(self)
        register_fork_reset(self)

    def _reset_after_fork_in_child(self) -> None:
        """Repair a forked child: fresh locks/client and cleared loop state.

        The parent's event loop, flush task, and breaker task do not exist in the
        child; clear them so ``_ensure_started`` relaunches the flush loop in the
        child's own loop on the next enqueue. The inherited client is abandoned
        (never closed — the parent owns those sockets). Queued spend items
        duplicated into the child by fork are deliberately KEPT: the server
        dedups them. Inherited advisory observations are discarded because
        parent and child mint different report IDs.
        """
        self._breaker_project_lock = threading.Lock()
        self._breaker_report_lock = threading.Lock()
        if self._untracked_state is not None:
            self._untracked_state.reset_after_fork_in_child()
        self._receipt_fold_state.reset_after_fork_in_child()
        self._drop_lock = threading.Lock()
        self._ownership_lock = threading.Lock()
        self._in_hand = {}
        self._next_event_batch_token = 0
        self._event_batches_in_hand = {}
        self._next_settlement_token = 0
        self._settlements_in_hand = {}
        self._in_flight = 0
        self._final_delivery_started = False
        self._flush_task = None
        self._breaker_task = None
        self._untracked_task = None
        self._shutdown_event = None
        self._http = httpx.AsyncClient(timeout=10.0)

    def start(self) -> None:
        """Start the background flush loop.  Must be called within an event loop.

        Idempotent: if a flush task is already running, this is a no-op and the
        existing ``_shutdown_event`` is left untouched (a second call must not
        orphan the live task or reset its shutdown signal). Restarting a closed
        reporter is a programming error and raises — the sync reporter is live
        from construction, so a closed async reporter has no valid restart.
        """
        if self._closed:
            raise RuntimeError("cannot start a closed AsyncMetadataReporter")
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._shutdown_event = asyncio.Event()
        self._flush_task = asyncio.create_task(self._flush_loop())

    def _notify_untracked_observation(self) -> None:
        """Start/wake advisory delivery without awaiting or doing network I/O."""
        if self._untracked_state is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Advisory state is retained for close()/GC/atexit. Unlike spend
            # events, it does not need to consume the no-loop warning latch.
            return
        try:
            self._ensure_started()
            if self._flush_task is None or self._flush_task.done():
                return
            if self._untracked_reports_due():
                self._start_untracked_cycle()
        except Exception:
            # Advisory activation must never affect the provider call.
            return

    def _ensure_started(self) -> None:
        """Auto-start the flush loop on first enqueue, when a loop is running.

        The sync reporter starts its flush thread in ``__init__``; the async
        reporter cannot (there may be no event loop yet), so it starts lazily on
        the first enqueue instead of only via ``start()`` / ``__aenter__``.
        Without this, a reporter constructed outside ``async with`` queues
        events AND budget-confirm settlements silently until ``close()``, so
        server-side spend tracking drifts.

        Never raises (called on the enqueue path): a closed reporter, an
        already-running flush task, or the absence of a running loop each
        short-circuit. With no running loop the event stays queued and a single
        warning per reporter instance is logged.
        """
        if self._closed:
            return
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if not self._warned_no_loop:
                self._warned_no_loop = True
                _log_warning(
                    "reporter.enqueue_without_event_loop: no running event loop; "
                    "events stay queued until start() or close() runs inside a loop"
                )
            return
        self.start()

    def report(self, event: MetadataEvent) -> None:
        """Enqueue a metadata event for async reporting.  Non-blocking."""
        if self._closed:
            self._fold_or_count_event_drop(event, "closed_enqueue")
            return
        self._ensure_started()
        # Ownership-gated: a producer thread that passed the _closed check and
        # then lost the race with close()'s seal must be refused-and-counted,
        # never appended into a queue nothing will ever drain (#10 review pin).
        if not self._enqueue(event):
            self._fold_or_count_event_drop(event, "closed_enqueue")

    def report_confirm(self, request: BudgetConfirmRequest) -> None:
        """Fire-and-forget a confirm request onto the async flush queue."""
        if self._closed:
            self._count_drop("confirm", "closed_enqueue")
            return
        self._ensure_started()
        enqueued = self._enqueue_owned(
            self._confirm_queue, _PendingConfirm(request), _MAX_PENDING_CONTROL, "confirm"
        )
        if not enqueued:
            self._count_drop("confirm", "closed_enqueue")

    def report_settlement(self, request: BudgetConfirmRequest, event: MetadataEvent) -> None:
        """Fire-and-forget a stream settlement as one ordered queue item."""
        if self._closed:
            # The pair loses its confirm AND its event — count both halves.
            self._count_drop("settlement_confirm", "closed_enqueue")
            self._fold_or_count_event_drop(event, "closed_enqueue")
            return
        self._ensure_started()
        enqueued = self._enqueue_settlement_owned(
            _PendingSettlement(_PendingConfirm(request), event)
        )
        if not enqueued:
            self._count_drop("settlement_confirm", "closed_enqueue")
            self._fold_or_count_event_drop(event, "closed_enqueue")

    def _begin_blocking_exit(self) -> None:
        """Refuse public work while lifecycle drains existing queues.

        The blocking atexit path invokes user logging callbacks after its last
        queue sweep. A callback may reenter ``report*``; closing both public
        and ownership enqueue gates before the drain ensures that work gets a
        terminal disposition instead of landing in an ownerless post-exit
        queue. Lifecycle still consumes the already-queued items directly.
        """
        with self._ownership_lock:
            self._closed = True
            self._delivery_closed = True

    async def close(self, timeout: float | None = None) -> None:
        """Flush remaining events and shut down within a single deadline.

        See ``MetadataReporter.close`` — one monotonic ``deadline`` bounds the
        flush-task await, the final flush, and the breaker report cycle.

        ``_closed`` (no new work) is set before the first await; the exit
        rescue state (``_close_completed`` + the GC finalizer) is touched only
        AFTER the flush chain finishes. A close() cancelled mid-await
        propagates the cancellation but leaves both lifecycle rescue paths
        armed for whatever spend is still queued.
        """
        self._closed = True
        budget = self.shutdown_deadline if timeout is None else timeout
        deadline = _monotonic() + budget
        active_breaker_task = self._breaker_task
        if active_breaker_task is not None and active_breaker_task.done():
            active_breaker_task = None
        active_untracked_task = self._untracked_task
        if active_untracked_task is not None and active_untracked_task.done():
            active_untracked_task = None
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._flush_task is not None:
            await self._await_within(self._flush_task, deadline)
        # Take terminal fold ownership before scheduling the final task. A
        # zero remaining deadline can cancel that task before its coroutine
        # body starts; folds must still have an owner and counted disposition.
        self._drain_receipt_folds_to_queue(final=True)
        # The final flush rides its own task so the shared deadline can CANCEL
        # a slow-drip send: httpx timeouts bound individual socket operations,
        # not total response time, so an inline await could outlive the
        # deadline indefinitely (#4 review pin). A cancelled drain requeues its
        # in-hand item before re-raising — the seal below counts it, or a
        # cancelled close leaves it queued for the lifecycle rescue.
        await self._await_within(
            asyncio.ensure_future(self._flush_remaining(deadline=deadline, final=True)), deadline
        )
        # Seal delivery: refuse enqueues from here on and claim any straggler a
        # producer thread appended after the final drain passed its queue
        # (#10 review pin). Skipped if close() was cancelled above — a
        # cancelled close leaves delivery open for the lifecycle rescue paths.
        self._seal_delivery()

        if active_untracked_task is not None:
            await self._await_within(active_untracked_task, deadline)
        if not self._deadline_expired(deadline) and self._untracked_reports_due():
            final_untracked_task = self._start_untracked_cycle(
                during_shutdown=True,
                deadline=deadline,
            )
            if final_untracked_task is not None:
                await self._await_within(final_untracked_task, deadline)
        if active_breaker_task is not None:
            await self._await_within(active_breaker_task, deadline)
        if active_breaker_task is None or not self._deadline_expired(deadline):
            final_breaker_task = self._start_breaker_cycle(
                during_shutdown=True,
                deadline=deadline,
                force=True,
            )
            if final_breaker_task is not None:
                await self._await_within(final_breaker_task, deadline)
        await self._http.aclose()
        # A COMPLETED close supersedes the exit-flush safety nets: drop the GC
        # finalizer so it can never double-flush drained queues, and tell the
        # atexit hook this reporter needs no rescue.
        self._close_completed = True
        if self._finalizer is not None:
            self._finalizer.detach()

    async def _await_within(self, task: asyncio.Task[None], deadline: float) -> None:
        """Await ``task`` but never past the shared shutdown deadline.

        On timeout the task is cancelled — during close we would rather abandon a
        stuck flush/breaker task than exceed the deadline; a duplicate send is
        server-deduped.
        """
        remaining = deadline - _monotonic()
        if remaining <= 0:
            # Deadline already spent: cancel, then STILL await the task so its
            # cleanup — the drains' CancelledError requeue — returns every
            # in-hand item to its queue before close() proceeds to seal and
            # detach the exit rescue (#7 review pin). wait_for below gives the timeout
            # path the same guarantee (it awaits the cancelled task before
            # raising), so only this fast path needed it.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            return
        try:
            await asyncio.wait_for(task, timeout=remaining)
        except TimeoutError:
            pass  # wait_for already cancelled (and awaited) the stuck task
        except asyncio.CancelledError:
            # Two sources are conflated here: close() ITSELF being cancelled
            # (must propagate — a cancelled close must not silently keep
            # running) vs the awaited task having been cancelled elsewhere
            # (safe to continue shutting down).
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise

    async def __aenter__(self) -> AsyncMetadataReporter:
        self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _flush_loop(self) -> None:
        """Background task: periodically flush batches to the cloud."""
        if self._shutdown_event is None:
            raise RuntimeError("_flush_loop called before reporter was started")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.flush_interval,
                )
            except TimeoutError:
                try:
                    await self._flush_remaining()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # The flush task must survive anything a drain raises: a
                    # dead flush task strands ALL queued spend until overflow.
                    _log_warning("reporter.flush_cycle_failed: exc_type=%s", type(exc).__name__)
                if self._breaker_reports_due():
                    self._start_breaker_cycle()
                if self._untracked_reports_due():
                    self._start_untracked_cycle()
            else:
                break

    async def _flush_remaining(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Flush queued confirms, settlements, then metadata events in batches.

        See ``MetadataReporter._flush_remaining`` for the ``final`` / ``deadline``
        semantics.
        """
        self._drain_receipt_folds_to_queue(final=final)
        await self._drain_confirms(deadline=deadline, final=final)
        await self._drain_settlements(deadline=deadline, final=final)
        if await self._drain_event_batches(deadline=deadline, final=final):
            self._receipt_fold_state.note_cycle_success()
        self._maybe_log_drops(force=final)

    async def _drain_confirms(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Send due confirms in FIFO order; a retrying head parks the queue."""
        while self._confirm_queue:
            if self._deadline_expired(deadline):
                self._count_drop(
                    "confirm", "shutdown_deadline", n=_drain_count(self._confirm_queue)
                )
                break
            try:
                # Atomic pop, never peek-then-pop: a cancelled close() and the
                # cooperatively-scheduled flush task can interleave drains.
                pending = self._confirm_queue.popleft()
            except IndexError:
                break
            if not final and pending.next_attempt_at > _monotonic():
                self._confirm_queue.appendleft(pending)
                break
            try:
                outcome = await self._send_confirm(
                    pending.request, timeout=self._send_timeout(deadline)
                )
            except asyncio.CancelledError:
                # A cancelled drain's in-hand item is still deliverable: a
                # cancelled close() leaves rescue armed, and rescue only sees
                # the QUEUES — so requeue for it (#4 re-review pin). A
                # completing close seals next and counts the requeued item;
                # only a seal-refused requeue is counted here.
                if not self._requeue_cancelled(self._confirm_queue, [pending]):
                    self._count_drop("confirm", "shutdown_deadline")
                raise
            if outcome is _SendOutcome.SENT:
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    self._count_drop(
                        "confirm", "exit_breaker_open", n=1 + _drain_count(self._confirm_queue)
                    )
                else:
                    self._confirm_queue.appendleft(pending)
                break
            if outcome is _SendOutcome.RETRY:
                pending.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.attempts)
                if finished or final:
                    self._count_drop("confirm", "retry_exhausted")
                    continue
                pending.next_attempt_at = next_at
                self._confirm_queue.appendleft(pending)
                break  # FIFO: nothing behind a backing-off head may jump it
            self._count_drop("confirm", "terminal_status")

    async def _drain_settlements(
        self, *, deadline: float | None = None, final: bool = False
    ) -> None:
        """Resolve each settlement's confirm first, then hand its event to the
        event queue (confirm-before-metadata order per item is load-bearing).
        A retrying head parks the queue behind it (FIFO)."""
        while self._settlement_queue:
            if self._deadline_expired(deadline):
                stranded: list[_PendingSettlement] = []
                while self._settlement_queue:
                    stranded.append(self._settlement_queue.popleft())
                self._dispose_settlements(stranded, "shutdown_deadline")
                break
            try:
                # Atomic pop — see _drain_confirms for the interleaving
                # rationale.
                pending = self._settlement_queue.popleft()
            except IndexError:
                break
            if not final and pending.confirm.next_attempt_at > _monotonic():
                self._settlement_queue.appendleft(pending)
                break
            try:
                outcome = await self._send_confirm(
                    pending.confirm.request, timeout=self._send_timeout(deadline)
                )
            except asyncio.CancelledError:
                # Same rescue-requeue as _drain_confirms; a seal-refused pair
                # loses its confirm AND its event — count both.
                if not self._requeue_cancelled(self._settlement_queue, [pending]):
                    self._count_drop("settlement_confirm", "shutdown_deadline")
                    self._fold_or_count_event_drop(pending.event, "shutdown_deadline")
                raise
            if outcome is _SendOutcome.SENT:
                self._move_event_to_queue(pending.event)
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    # The confirms are undeliverable (control plane known-down),
                    # but ingest is NOT breaker-gated: hand every event to the
                    # event drain so the durable spend truth still gets its
                    # deadline-bounded exit attempt.
                    stranded = [pending, *self._settlement_queue]
                    self._settlement_queue.clear()
                    self._count_drop("settlement_confirm", "exit_breaker_open", n=len(stranded))
                    for orphan in stranded:
                        self._move_event_to_queue(orphan.event)
                else:
                    self._settlement_queue.appendleft(pending)
                break
            if outcome is _SendOutcome.RETRY:
                pending.confirm.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.confirm.attempts)
                if finished or final:
                    self._count_drop("settlement_confirm", "retry_exhausted")
                    self._move_event_to_queue(pending.event)
                    continue
                pending.confirm.next_attempt_at = next_at
                self._settlement_queue.appendleft(pending)
                break
            self._count_drop("settlement_confirm", "terminal_status")
            self._move_event_to_queue(pending.event)

    async def _drain_event_batches(
        self, *, deadline: float | None = None, final: bool = False
    ) -> bool:
        """Send due metadata events in batches; requeue a failed batch to front."""
        sent_clean_batch = False
        cycle_clean = True
        while self._queue:
            if self._deadline_expired(deadline):
                cycle_clean = False
                stranded: list[_PendingEvent] = []
                while self._queue:
                    stranded.append(self._queue.popleft())
                self._dispose_pending_events(stranded, "shutdown_deadline")
                break
            if self._in_flight >= self.max_in_flight:
                break
            now = _monotonic()
            prefix: list[_PendingEvent] = []
            while len(prefix) < self.batch_size and self._queue:
                head = self._queue[0]
                if not final and head.next_attempt_at > now:
                    break
                prefix.append(self._queue.popleft())
            if not prefix:
                break
            try:
                result = await self._send_batch([p.event for p in prefix], deadline=deadline)
            except asyncio.CancelledError:
                # Same rescue-requeue as _drain_confirms, for the whole batch.
                if not self._requeue_cancelled(self._queue, prefix):
                    self._dispose_pending_events(prefix, "shutdown_deadline")
                raise
            if result.outcome is _SendOutcome.SENT:
                self._publish_ingest_rejections(
                    [pending.event for pending in prefix], result.rejections
                )
                clean = result.rejections.kind is _IngestRejectionKind.CLEAN
                sent_clean_batch = clean or sent_clean_batch
                cycle_clean = clean and cycle_clean
                continue
            if result.outcome is _SendOutcome.RETRY:
                cycle_clean = False
                keep: list[_PendingEvent] = []
                dropped: list[_PendingEvent] = []
                for p in prefix:
                    p.attempts += 1
                    finished, next_at = self._resolve_retryable(attempts=p.attempts)
                    if finished or final:
                        dropped.append(p)
                    else:
                        p.next_attempt_at = next_at
                        keep.append(p)
                self._dispose_pending_events(dropped, "retry_exhausted")
                if keep:
                    self._queue.extendleft(reversed(keep))
                break
            cycle_clean = False
            self._dispose_pending_events(prefix, "terminal_status")
        return sent_clean_batch and cycle_clean

    def _start_breaker_cycle(
        self,
        *,
        during_shutdown: bool = False,
        deadline: float | None = None,
        force: bool = False,
    ) -> asyncio.Task[None] | None:
        """Start one tracked breaker task, or coalesce into the active task."""
        if (
            self._shutdown_event is not None
            and self._shutdown_event.is_set()
            and not during_shutdown
        ):
            return None
        task = self._breaker_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(
            self._flush_breaker_reports(deadline=deadline, force=force),
            name="solwyn-breaker-reporter",
        )
        self._breaker_task = task
        return task

    async def _flush_breaker_reports(
        self, deadline: float | None = None, *, force: bool = False
    ) -> None:
        """POST current breaker snapshots independently and drop every failure.

        Bounded by the shared shutdown ``deadline`` when set (advisory snapshots
        are dropped, not counted as spend).
        """
        for project_id, report in self._build_breaker_reports(force=force):
            if self._deadline_expired(deadline):
                return
            try:
                response = await self._http.post(
                    f"{self.api_url}/api/v1/projects/{project_id}/providers/breaker-reports",
                    json=report.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=self._send_timeout(deadline),
                )
                response.raise_for_status()
                with self._breaker_report_lock:
                    self._breaker_last_sent[report.provider.value] = (
                        report.state,
                        report.failure_count,
                        report.success_count,
                    )
            except Exception as exc:
                if handle_read_only_key_error(exc):
                    # A read-only key denies every write: end the cycle instead
                    # of posting the remaining doomed snapshots.
                    return
                _log_warning(
                    "reporter.breaker_send_failed: provider=%s exc_type=%s",
                    report.provider.value,
                    type(exc).__name__,
                )

    def _start_untracked_cycle(
        self,
        *,
        during_shutdown: bool = False,
        deadline: float | None = None,
    ) -> asyncio.Task[None] | None:
        """Start one advisory report task, coalescing concurrent flush ticks."""
        if self._untracked_state is None:
            return None
        if (
            self._shutdown_event is not None
            and self._shutdown_event.is_set()
            and not during_shutdown
        ) or self._deadline_expired(deadline):
            return None
        task = self._untracked_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(
            self._flush_untracked_reports(deadline=deadline),
            name="solwyn-untracked-surface-reporter",
        )

        def _continue_if_due(_completed: asyncio.Task[None]) -> None:
            if self._untracked_task is _completed:
                self._untracked_task = None
            if (
                not (self._shutdown_event is not None and self._shutdown_event.is_set())
                and self._untracked_reports_due()
            ):
                self._start_untracked_cycle()

        task.add_done_callback(_continue_if_due)
        self._untracked_task = task
        return task

    async def _flush_untracked_reports(self, *, deadline: float | None = None) -> None:
        """POST eligible deltas in advisory batches and silently drop failures."""
        built_reports = self._build_untracked_reports()
        for offset in range(0, len(built_reports), _UNTRACKED_REPORT_BATCH_SIZE):
            if self._deadline_expired(deadline):
                return
            batch = built_reports[offset : offset + _UNTRACKED_REPORT_BATCH_SIZE]
            if not batch:
                continue
            self._mark_untracked_reports_attempted(batch)
            try:
                response = await self._http.post(
                    f"{self.api_url}/api/v1/untracked-surfaces",
                    json=[built.report.model_dump(mode="json") for built in batch],
                    headers=self._auth_headers(),
                    timeout=10.0 if deadline is None else self._send_timeout(deadline),
                )
                response.raise_for_status()
            except Exception:
                continue
            self._mark_untracked_reports_sent(batch)

    async def _send_confirm(
        self, confirm_request: BudgetConfirmRequest, *, timeout: float = 5.0
    ) -> _SendOutcome:
        """Send one confirm request and return its delivery outcome.

        See ``MetadataReporter._send_confirm``.
        """
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            _log_debug("reporter.confirm_held_breaker_open")
            return _SendOutcome.HELD
        try:
            resp = await self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=confirm_request.model_dump(mode="json"),
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            if breaker is not None:
                breaker.record_success()
            self._record_confirm_success()
            return _SendOutcome.SENT
        except Exception as exc:
            if handle_read_only_key_error(exc):
                if breaker is not None:
                    breaker.record_success()
                return _SendOutcome.SENT
            if breaker is not None:
                breaker.record_failure()
            self._record_confirm_failure(exc)
            return self._confirm_failure_outcome(exc)
        finally:
            # Cancellation (or any BaseException) bypasses the handler above; a
            # consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a success/failure verdict
            # has already released the slot.
            if breaker is not None:
                breaker.release_probe(admission)

    async def _send_batch(
        self, batch: list[MetadataEvent], *, deadline: float | None = None
    ) -> _BatchSendResult:
        """Send a batch and return transport plus parsed member dispositions.

        Ingest is deliberately NOT control-plane-breaker-guarded: opening the
        enforcement breaker (which flips budget checks to their fail-open
        posture) on an ingest blip would be a worse failure mode than a delayed
        batch. Ingest self-paces via the retry backoff instead.

        ``deadline`` clamps the request into the shutdown window; without it a
        black-holed control plane made close() overrun its budget by the full
        10s client timeout (P0 review finding).
        """
        timeout = 10.0 if deadline is None else self._send_timeout(deadline)
        self._in_flight += 1
        try:
            payload = [e.model_dump(mode="json") for e in batch]
            resp = await self._http.post(
                f"{self.api_url}/api/v1/metadata/ingest",
                json=payload,
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            # httpx.Response.json() is sync on both clients — shared helper.
            rejections = self._parse_ingest_rejections(resp, len(batch))
            return _BatchSendResult(_SendOutcome.SENT, rejections)
        except Exception as exc:
            if handle_read_only_key_error(exc):
                return _BatchSendResult(
                    _SendOutcome.SENT, _IngestRejections(_IngestRejectionKind.MALFORMED)
                )
            # Log only the exception's class name — the type-name-only
            # convention every other except-block follows. Safe here even though
            # the batch is content-free; keeps the privacy contract uniform.
            _log_warning(
                "Failed to send metadata batch (%d events): %s",
                len(batch),
                type(exc).__name__,
            )
            return _BatchSendResult(
                self._batch_failure_outcome(exc),
                _IngestRejections(_IngestRejectionKind.MALFORMED),
            )
        finally:
            self._in_flight -= 1
