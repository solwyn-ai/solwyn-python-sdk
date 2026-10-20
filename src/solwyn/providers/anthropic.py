"""Anthropic provider adapter — token extraction only, no pricing.

Anthropic usage normalization:

    input_tokens (normalized) = base_input + cache_read + cache_5m + cache_1h

Anthropic reports cache_read_input_tokens as a separate additive field (not a
subset of input_tokens). Cache writes are broken out by TTL tier in the
usage.cache_creation sub-object:

    usage.cache_creation.ephemeral_5m_input_tokens  — 5-minute TTL (1.25× rate)
    usage.cache_creation.ephemeral_1h_input_tokens  — 1-hour TTL  (2×    rate)

When the cache_creation sub-object is absent but cache_creation_input_tokens is
present, the aggregate is attributed to the 5m bucket because that is
Anthropic's default prompt-cache TTL.

reasoning_tokens stays 0 — Anthropic folds extended thinking tokens into
output_tokens and does not report them separately (documented blind spot).

Server-side compaction (beta) is billed but reported ONLY in
``usage.iterations``: each ``type == "compaction"`` entry carries its own
input/output/cache counts that the top-level usage fields exclude. Those sums
are added INTO the parent buckets (same normalization as the top level) and
also reported as the ``compaction_*`` subsets. Other iteration types
(``message``, ``advisor_message``, ``fallback_message``) are already covered by
the top-level fields and are ignored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from solwyn._constants import ORDINARY_TOKEN_COUNT_MAX
from solwyn._token_details import TokenDetails

logger = logging.getLogger(__name__)


def _parse_anthropic_cache(usage: object) -> tuple[int, int]:
    """Return (5m cache-write tokens, 1h cache-write tokens) from Anthropic usage."""
    detail = getattr(usage, "cache_creation", None)
    if detail is not None:
        cache_5m = getattr(detail, "ephemeral_5m_input_tokens", None) or 0
        cache_1h = getattr(detail, "ephemeral_1h_input_tokens", None) or 0
    else:
        cache_5m = getattr(usage, "cache_creation_input_tokens", None) or 0
        cache_1h = 0
    return cache_5m, cache_1h


_MISSING = object()


def _read(obj: object, name: str) -> object:
    """Read one usage attribute from a pydantic/namespace object or a plain dict.

    ``iterations`` is a typed list on beta responses but an untyped extra (a
    list of plain dicts) when compaction is requested through ``extra_body``
    on the non-beta surface.
    """
    if isinstance(obj, Mapping):
        return obj.get(name, _MISSING)
    return getattr(obj, name, _MISSING)


def _count(value: object) -> int | None:
    """A valid token count, 0 for an absent one, or None when malformed."""
    if value is _MISSING or value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass(frozen=True)
class _CompactionUsage:
    """Summed usage of the ``compaction`` entries in ``usage.iterations``."""

    base_input: int = 0
    cache_read: int = 0
    cache_5m: int = 0
    cache_1h: int = 0
    output: int = 0

    @property
    def input_total(self) -> int:
        return self.base_input + self.cache_read + self.cache_5m + self.cache_1h


def _compaction_entry(entry: object) -> _CompactionUsage | None:
    """Parse one compaction iteration; None when any count is malformed.

    The cache split mirrors ``_parse_anthropic_cache``: the per-TTL
    ``cache_creation`` breakdown when present, else the aggregate
    ``cache_creation_input_tokens`` attributed to the default 5m tier.
    """
    base_input = _count(_read(entry, "input_tokens"))
    output = _count(_read(entry, "output_tokens"))
    cache_read = _count(_read(entry, "cache_read_input_tokens"))
    detail = _read(entry, "cache_creation")
    if detail is not _MISSING and detail is not None:
        cache_5m = _count(_read(detail, "ephemeral_5m_input_tokens"))
        cache_1h = _count(_read(detail, "ephemeral_1h_input_tokens"))
    else:
        cache_5m = _count(_read(entry, "cache_creation_input_tokens"))
        cache_1h = 0
    if base_input is None or output is None or cache_read is None:
        return None
    if cache_5m is None or cache_1h is None:
        return None
    return _CompactionUsage(base_input, cache_read, cache_5m, cache_1h, output)


def _read_iterations(usage: object) -> object:
    """Return ``usage.iterations`` when present and a list; otherwise None."""
    if usage is None:
        return None
    iterations = _read(usage, "iterations")
    return iterations if isinstance(iterations, list) else None


def _sum_compaction(iterations: object) -> _CompactionUsage:
    """Sum every well-formed ``type == "compaction"`` entry. Never raises."""
    if not isinstance(iterations, list):
        return _CompactionUsage()
    total = _CompactionUsage()
    for entry in iterations:
        if _read(entry, "type") != "compaction":
            continue
        parsed = _compaction_entry(entry)
        if parsed is None:
            continue
        total = _CompactionUsage(
            total.base_input + parsed.base_input,
            total.cache_read + parsed.cache_read,
            total.cache_5m + parsed.cache_5m,
            total.cache_1h + parsed.cache_1h,
            total.output + parsed.output,
        )
    return total


def _token_details(
    *,
    base_input: int,
    cache_read: int,
    cache_5m: int,
    cache_1h: int,
    output: int,
    compaction: _CompactionUsage,
) -> TokenDetails:
    """Build normalized TokenDetails with compaction folded into the parents.

    Each ``compaction_*`` subset is clamped to the wire bound and to its
    parent bucket, and the compaction cache subsets to the clamped
    ``compaction_input_tokens``, so the model's subset validator can never
    fail on provider data.
    """
    input_tokens = base_input + cache_read + cache_5m + cache_1h + compaction.input_total
    cached = cache_read + compaction.cache_read
    parent_5m = cache_5m + compaction.cache_5m
    parent_1h = cache_1h + compaction.cache_1h
    output_tokens = output + compaction.output

    compaction_input = min(compaction.input_total, input_tokens, ORDINARY_TOKEN_COUNT_MAX)
    compaction_cached = min(compaction.cache_read, cached, compaction_input)
    compaction_5m = min(compaction.cache_5m, parent_5m, compaction_input - compaction_cached)
    compaction_1h = min(
        compaction.cache_1h, parent_1h, compaction_input - compaction_cached - compaction_5m
    )
    return TokenDetails(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        cache_creation_5m_tokens=parent_5m,
        cache_creation_1h_tokens=parent_1h,
        compaction_input_tokens=compaction_input,
        compaction_cached_input_tokens=compaction_cached,
        compaction_cache_creation_5m_tokens=compaction_5m,
        compaction_cache_creation_1h_tokens=compaction_1h,
        compaction_output_tokens=min(compaction.output, output_tokens, ORDINARY_TOKEN_COUNT_MAX),
    )


class AnthropicAdapter:
    """Extracts normalized TokenDetails from Anthropic API responses."""

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def dialect(self) -> Literal["anthropic"]:
        return "anthropic"

    def detect_client(self, client: Any) -> bool:
        """Return True if the client's module path contains 'anthropic'."""
        return "anthropic" in getattr(type(client), "__module__", "")

    def detect_model(self, model: str) -> bool:
        """Return True for claude-* model prefix."""
        return model.startswith("claude-")

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract token usage from an Anthropic messages.create() response.

        Returns TokenDetails() with all zeros when usage is unavailable.
        Never raises — returns zeros for any unexpected response shape.

        Normalization:
        - input_tokens = base + cache_read + (5m + 1h cache writes)  (all additive)
        - cached_input_tokens = cache_read_input_tokens
        - cache_creation_5m_tokens = usage.cache_creation.ephemeral_5m_input_tokens,
          or aggregate cache_creation_input_tokens when only the aggregate exists
        - cache_creation_1h_tokens = usage.cache_creation.ephemeral_1h_input_tokens
        - reasoning_tokens = 0 (folded into output_tokens by Anthropic)

        The cache_creation sub-object is absent on non-beta/older response shapes.
        Those responses can still carry cache_creation_input_tokens as an aggregate.

        Compaction iterations (``usage.iterations`` entries of type
        ``compaction``) are excluded from the top-level fields by Anthropic, so
        their normalized counts are added to the parent buckets and reported
        as the ``compaction_*`` subsets.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenDetails()

        base_input = getattr(usage, "input_tokens", None) or 0
        output = getattr(usage, "output_tokens", None) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_5m, cache_1h = _parse_anthropic_cache(usage)

        return _token_details(
            base_input=base_input,
            cache_read=cache_read,
            cache_5m=cache_5m,
            cache_1h=cache_1h,
            output=output,
            compaction=_sum_compaction(_read_iterations(usage)),
        )

    def estimate_missing_usage(
        self, response: Any, *, estimated_input_tokens: int
    ) -> TokenDetails | None:
        """Anthropic always reports usage — no estimated fallback."""
        return None

    def extract_service_tier(self, response: Any) -> str | None:
        """Anthropic responses do not expose a service tier."""
        return None

    def extract_region(self, client: Any) -> str | None:
        """Anthropic pricing is not regional."""
        return None

    def prepare_streaming(
        self, kwargs: dict[str, Any], *, cross_provider: bool = False
    ) -> dict[str, Any]:
        """Anthropic streams include usage events by default — no changes needed."""
        return dict(kwargs)

    def create_stream_accumulator(
        self, *, estimated_input_tokens: int = 0
    ) -> AnthropicStreamAccumulator:
        """Anthropic streams always carry usage events — the input estimate
        is not needed."""
        return AnthropicStreamAccumulator()

    def prepare_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Messages hop: streaming rides the ``stream=True`` kwarg.

        timeout/max_retries are ignored — the dispatcher already applied them
        via the SDK's ``with_options``.
        """
        kwargs = dict(kwargs)
        if is_streaming:
            kwargs["stream"] = True
        return client.messages.create, kwargs

    def prepare_beta_messages_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Beta Messages hop: same shape as ``prepare_call`` on ``client.beta``.

        Beta request fields (``betas``, ``context_management``,
        ``compaction``, ...) pass through untouched.
        """
        kwargs = dict(kwargs)
        if is_streaming:
            kwargs["stream"] = True
        return client.beta.messages.create, kwargs

    def unwrap_stream_source(self, response: Any) -> Any:
        """The streaming call returns the iterable itself."""
        return response

    def wrap_stream_result(self, wrapper: Any, served_response: Any) -> Any:
        """Anthropic-dialect callers iterate the stream object directly."""
        return wrapper


class AnthropicStreamAccumulator:
    """Accumulates usage from Anthropic streaming events.

    Input tokens arrive in message_start.message.usage.
    Output tokens arrive in message_delta.usage.
    Cache fields are on message_start and may be absent on older APIs.

    message_delta usage counts are CUMULATIVE totals: input/cache counts there
    (when present) take ``max(message_start, message_delta)`` per bucket, so
    an endpoint that omits or zeroes them never lowers the message_start
    value. message_delta carries only the aggregate cache_creation count; any
    growth past the message_start breakdown is attributed to the default 5m
    tier. ``usage.iterations`` is replaced (never summed) by the last
    message_delta that carries it, falling back to message_start's.
    """

    def __init__(self) -> None:
        self._base_input: int = 0
        self._cache_read: int = 0
        self._cache_5m: int = 0
        self._cache_1h: int = 0
        self._output: int = 0
        self._start_iterations: object = None
        self._delta_iterations: object = None
        self._saw_message_start = False
        self._saw_message_delta = False

    def observe(self, chunk: Any) -> None:
        event_type = getattr(chunk, "type", None)

        if event_type == "message_start":
            self._saw_message_start = True
            usage = getattr(getattr(chunk, "message", None), "usage", None)
            if usage is not None:
                self._base_input = getattr(usage, "input_tokens", None) or 0
                self._cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
                self._cache_5m, self._cache_1h = _parse_anthropic_cache(usage)
                self._start_iterations = _read_iterations(usage)

        elif event_type == "message_delta":
            self._saw_message_delta = True
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._output = getattr(usage, "output_tokens", None) or 0
                self._observe_cumulative_input(usage)
                iterations = _read_iterations(usage)
                if iterations is not None:
                    self._delta_iterations = iterations

    def _observe_cumulative_input(self, usage: object) -> None:
        base_input = _count(getattr(usage, "input_tokens", None))
        if base_input:
            self._base_input = max(self._base_input, base_input)
        cache_read = _count(getattr(usage, "cache_read_input_tokens", None))
        if cache_read:
            self._cache_read = max(self._cache_read, cache_read)
        cache_creation = _count(getattr(usage, "cache_creation_input_tokens", None))
        if cache_creation:
            growth = cache_creation - (self._cache_5m + self._cache_1h)
            if growth > 0:
                self._cache_5m += growth

    def finalize(self) -> TokenDetails:
        if self._output > 0 and not self._saw_message_start:
            logger.warning(
                "Anthropic stream finalized without message_start; "
                "input token counts may be incomplete"
            )
        input_total = self._base_input + self._cache_read + self._cache_5m + self._cache_1h
        if input_total > 0 and not self._saw_message_delta:
            logger.warning(
                "Anthropic stream finalized without message_delta; "
                "output token counts may be incomplete"
            )
        iterations = (
            self._delta_iterations if self._delta_iterations is not None else self._start_iterations
        )
        return _token_details(
            base_input=self._base_input,
            cache_read=self._cache_read,
            cache_5m=self._cache_5m,
            cache_1h=self._cache_1h,
            output=self._output,
            compaction=_sum_compaction(iterations),
        )

    def get_service_tier(self) -> str | None:
        """Anthropic streams do not expose a service tier."""
        return None
