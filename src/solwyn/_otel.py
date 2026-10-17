"""Optional, sans-I/O capture of the caller's active trace identifiers.

Importing Solwyn never imports OpenTelemetry. Only the span context is read;
the span itself, its attributes, events and propagation data are never retained.
"""

from __future__ import annotations

import re
from threading import Lock
from types import ModuleType
from typing import NamedTuple

from solwyn._constants import (
    SPAN_ID_HEX_LENGTH,
    SPAN_ID_PATTERN,
    TRACE_ID_HEX_LENGTH,
    TRACE_ID_PATTERN,
)

_trace_id_pattern = re.compile(TRACE_ID_PATTERN)
_span_id_pattern = re.compile(SPAN_ID_PATTERN)
_trace_api: ModuleType | None = None
_trace_api_resolved = False
_trace_api_lock = Lock()


def _get_trace_api() -> ModuleType | None:
    """Resolve the optional API once, including an unavailable or broken import."""
    global _trace_api, _trace_api_resolved
    if not _trace_api_resolved:
        with _trace_api_lock:
            if not _trace_api_resolved:
                try:
                    from opentelemetry import trace
                except Exception:
                    _trace_api = None
                else:
                    _trace_api = trace
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
