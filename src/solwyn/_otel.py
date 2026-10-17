"""Optional, sans-I/O capture of the caller's active trace identifiers.

Importing Solwyn never imports OpenTelemetry. Only the span context is read;
the span itself, its attributes, events and propagation data are never retained.
"""

from __future__ import annotations

from typing import NamedTuple


class _TraceSnapshot(NamedTuple):
    trace_id: str
    span_id: str
    trace_sampled: bool


def capture_trace() -> _TraceSnapshot | None:
    """Snapshot an existing valid span, or return None when tracing is idle."""
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return _TraceSnapshot(
            trace.format_trace_id(context.trace_id),
            trace.format_span_id(context.span_id),
            context.trace_flags.sampled,
        )
    except Exception:
        # Optional enrichment must not interrupt a call, including when an
        # application's custom span implementation cannot supply its context.
        return None
