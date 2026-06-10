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
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

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


class AnthropicAdapter:
    """Extracts normalized TokenDetails from Anthropic API responses."""

    @property
    def name(self) -> str:
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
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenDetails()

        base_input = getattr(usage, "input_tokens", None) or 0
        output = getattr(usage, "output_tokens", None) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_5m, cache_1h = _parse_anthropic_cache(usage)

        return TokenDetails(
            input_tokens=base_input + cache_read + cache_5m + cache_1h,
            output_tokens=output,
            cached_input_tokens=cache_read,
            cache_creation_5m_tokens=cache_5m,
            cache_creation_1h_tokens=cache_1h,
        )

    def extract_service_tier(self, response: Any) -> str | None:
        """Anthropic responses do not expose a service tier."""
        return None

    def extract_region(self, client: Any) -> str | None:
        """Anthropic pricing is not regional."""
        return None

    def prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Anthropic streams include usage events by default — no changes needed."""
        return dict(kwargs)

    def create_stream_accumulator(self) -> AnthropicStreamAccumulator:
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
    """

    def __init__(self) -> None:
        self._base_input: int = 0
        self._cache_read: int = 0
        self._cache_5m: int = 0
        self._cache_1h: int = 0
        self._output: int = 0
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

        elif event_type == "message_delta":
            self._saw_message_delta = True
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._output = getattr(usage, "output_tokens", None) or 0

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
        return TokenDetails(
            input_tokens=input_total,
            output_tokens=self._output,
            cached_input_tokens=self._cache_read,
            cache_creation_5m_tokens=self._cache_5m,
            cache_creation_1h_tokens=self._cache_1h,
        )

    def get_service_tier(self) -> str | None:
        """Anthropic streams do not expose a service tier."""
        return None
