"""Optional, sans-I/O trace correlation and content-free call spans.

Importing Solwyn never imports OpenTelemetry. Customer spans are read only via
their context; attributes and events are never read, even on our own spans.
"""

from __future__ import annotations

import os
import re
import weakref
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from threading import Lock, RLock
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, Self

from solwyn._constants import (
    SPAN_ID_HEX_LENGTH,
    SPAN_ID_PATTERN,
    TRACE_ID_HEX_LENGTH,
    TRACE_ID_PATTERN,
)
from solwyn.exceptions import RunStoppedError

if TYPE_CHECKING:
    from solwyn.budget import BudgetCheckResult

# Every emitter below uses literal keys. The AST firewall checks these literals
# against this closed set; adding a key is an explicit privacy-contract change.
CALL_SPAN_ATTRIBUTES = frozenset(
    {
        "solwyn.call_id",
        "solwyn.agent_run_id",
        "solwyn.parent_agent_run_id",
        "solwyn.budget.allowed",
        "solwyn.budget.mode",
        "solwyn.deny_source",
        "solwyn.denied_by_period",
        "solwyn.provider.requested",
        "solwyn.provider.served",
        "solwyn.model.requested",
        "solwyn.model.served",
        "solwyn.failover.reason",
        "solwyn.attempt_index",
        "solwyn.lease_funded",
    }
)

_trace_id_pattern = re.compile(TRACE_ID_PATTERN)
_span_id_pattern = re.compile(SPAN_ID_PATTERN)
_trace_api: ModuleType | None = None
_trace_api_resolved = False
_trace_api_lock = RLock()
_trace_api_loading = False


def _reset_after_fork() -> None:
    global _trace_api_lock, _trace_api_loading
    _trace_api_lock = RLock()
    _trace_api_loading = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _get_trace_api() -> ModuleType | None:
    """Resolve the optional API once, including an unavailable or broken import."""
    global _trace_api, _trace_api_resolved, _trace_api_loading
    if not _trace_api_resolved:
        with _trace_api_lock:
            if not _trace_api_resolved:
                # An import hook may re-enter a wrapped call on this thread.
                # Its enrichment is idle until the outer import has finished.
                if _trace_api_loading:
                    return None
                _trace_api_loading = True
                try:
                    from opentelemetry import trace
                except Exception:
                    _trace_api = None
                else:
                    _trace_api = trace
                finally:
                    _trace_api_loading = False
                _trace_api_resolved = True
    return _trace_api


class _TraceSnapshot(NamedTuple):
    trace_id: str
    span_id: str
    trace_sampled: bool


def _validated_trace(snapshot: _TraceSnapshot | None) -> _TraceSnapshot | None:
    """Discard malformed enrichment before it can invalidate a spend event."""
    if snapshot is None:
        return None
    try:
        trace_id, span_id, sampled = snapshot
        if (
            type(trace_id) is str
            and type(span_id) is str
            and type(sampled) is bool
            and _trace_id_pattern.fullmatch(trace_id)
            and _span_id_pattern.fullmatch(span_id)
            and trace_id != "0" * TRACE_ID_HEX_LENGTH
            and span_id != "0" * SPAN_ID_HEX_LENGTH
        ):
            return _TraceSnapshot(trace_id, span_id, sampled)
    except Exception:
        # A custom or otherwise malformed snapshot must also fail soft at the
        # event-building boundary; only plain strings and a real bool survive.
        return None
    return None


def capture_trace() -> _TraceSnapshot | None:
    """Snapshot an existing valid span, or return None when tracing is idle."""
    try:
        trace = _get_trace_api()
        if trace is None:
            return None
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return _validated_trace(
            _TraceSnapshot(
                trace.format_trace_id(context.trace_id),
                trace.format_span_id(context.span_id),
                context.trace_flags.sampled,
            )
        )
    except Exception:
        # Optional enrichment must not interrupt a call, including when an
        # application's custom span implementation cannot supply its context.
        return None


class _CallSpan:
    """One detached span, with explicit stream ownership and idempotent end.

    The inactive singleton has no mutable per-call state and performs no OTel
    work. All telemetry failures lose enrichment only, never a paid response.
    """

    def __init__(self, span: Any = None, trace: ModuleType | None = None) -> None:
        self._span = span
        self._trace = trace
        self._lock = Lock()
        self._ended = False
        self._deferred = False
        self._denied = False
        self._failover_reason: str | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, typ: object, exc: BaseException | None, tb: object) -> None:
        if exc is not None or not self._deferred:
            self.end(error=exc is not None, exception=exc)

    @contextmanager
    def current(self) -> Iterator[None]:
        """Attach only for preflight/dispatch; never across a stream return."""
        if self._span is None:
            yield
            return
        stack = ExitStack()
        if self._span is not None and self._trace is not None:
            with suppress(Exception):
                # Positional API flags: end_on_exit, exception recording, and
                # automatic error descriptions are ALL disabled for privacy.
                stack.enter_context(self._trace.use_span(self._span, False, False, False))
        try:
            yield
        finally:
            with suppress(Exception):
                stack.close()

    def capture_trace(self) -> _TraceSnapshot | None:
        # Exactly one snapshot, on the caller thread. An inactive span keeps the
        # caller's original context, including an existing non-recording parent.
        if self._span is None:
            return capture_trace()
        with self.current():
            return capture_trace()

    def identify(
        self,
        *,
        call_id: str,
        agent_run_id: str | None,
        parent_agent_run_id: str | None,
        provider: str,
        model: str,
        budget_mode: str,
    ) -> None:
        if self._span is None:
            return
        try:
            self._span.set_attribute("solwyn.call_id", call_id)
            self._span.set_attribute("solwyn.provider.requested", provider)
            self._span.set_attribute("solwyn.model.requested", model)
            self._span.set_attribute("solwyn.budget.mode", budget_mode)
            if agent_run_id is not None:
                self._span.set_attribute("solwyn.agent_run_id", agent_run_id)
            if parent_agent_run_id is not None:
                self._span.set_attribute("solwyn.parent_agent_run_id", parent_agent_run_id)
        except Exception:
            pass

    def budget(self, budget: BudgetCheckResult) -> None:
        if self._span is None:
            return
        try:
            self._denied = not budget.allowed
            self._span.set_attribute("solwyn.budget.allowed", budget.allowed)
            self._span.set_attribute("solwyn.budget.mode", budget.mode.value)
            self._span.set_attribute("solwyn.lease_funded", bool(getattr(budget, "lease_id", None)))
            if budget.deny_source is not None:
                self._span.set_attribute("solwyn.deny_source", budget.deny_source)
            if budget.denied_by_period is not None:
                self._span.set_attribute("solwyn.denied_by_period", budget.denied_by_period)
        except Exception:
            pass

    def denied(self, *, source: str, period: str) -> None:
        if self._span is None:
            return
        try:
            self._denied = True
            self._span.set_attribute("solwyn.budget.allowed", False)
            self._span.set_attribute("solwyn.deny_source", source)
            self._span.set_attribute("solwyn.denied_by_period", period)
        except Exception:
            pass

    def attempt(self, provider: str, model: str, index: int, reason: str | None = None) -> None:
        if self._span is None:
            return
        # Routing can return to the primary after a cost-routed hop fails. Defer
        # the optional reason until the terminal hop is known, so no stale key
        # survives (the portable OTel API has no attribute deletion operation).
        self._failover_reason = reason
        try:
            self._span.set_attribute("solwyn.provider.served", provider)
            self._span.set_attribute("solwyn.model.served", model)
            self._span.set_attribute("solwyn.attempt_index", index)
        except Exception:
            pass

    def defer(self) -> None:
        if self._span is not None:
            self._deferred = True

    def finalize_on_gc(self, owner: object) -> weakref.finalize[..., object] | None:
        if self._span is None:
            return None
        finalizer = weakref.finalize(owner, self.end, error=True)
        finalizer.atexit = False  # type: ignore[misc]
        return finalizer

    def end(self, *, error: bool = False, exception: BaseException | None = None) -> None:
        if self._span is None or self._trace is None:
            return
        with self._lock:
            if self._ended:
                return
            self._ended = True
        try:
            if self._failover_reason is not None:
                self._span.set_attribute("solwyn.failover.reason", self._failover_reason)
            if isinstance(exception, RunStoppedError):
                error = True
                if not self._denied:
                    self.denied(source="run_terminated", period="run_stopped")
            self._span.set_status(
                self._trace.StatusCode.ERROR if error else self._trace.StatusCode.OK
            )
        except Exception:
            pass
        finally:
            with suppress(Exception):
                self._span.end()


_INACTIVE_CALL_SPAN = _CallSpan()


def _start_call_span() -> _CallSpan:
    """Create a recording span only; never attach an unexported child ID."""
    span = None
    try:
        trace = _get_trace_api()
        if trace is None:
            return _INACTIVE_CALL_SPAN
        span = trace.get_tracer("solwyn").start_span("solwyn.call", kind=trace.SpanKind.INTERNAL)
        context = span.get_span_context()
        if span.is_recording() and context.is_valid and context.trace_flags.sampled:
            return _CallSpan(span, trace)
    except Exception:
        pass
    if span is not None:
        with suppress(Exception):
            span.end()
    return _INACTIVE_CALL_SPAN
