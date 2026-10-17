"""Optional, sans-I/O trace correlation and content-free call/run spans.

Importing Solwyn never imports OpenTelemetry. Customer spans are read only via
their context; attributes and events are never read, even on our own spans.
"""

from __future__ import annotations

import os
import re
import weakref
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
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
    from solwyn._base import _SolwynBase
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
RUN_SPAN_ATTRIBUTES = frozenset({"solwyn.agent_run_id", "solwyn.parent_agent_run_id"})
SPAN_ATTRIBUTES = CALL_SPAN_ATTRIBUTES | RUN_SPAN_ATTRIBUTES
SPAN_NAMES = frozenset({"solwyn.call", "solwyn.run"})

_trace_id_pattern = re.compile(TRACE_ID_PATTERN)
_span_id_pattern = re.compile(SPAN_ID_PATTERN)
_trace_api: ModuleType | None = None
_trace_api_resolved = False
_trace_api_lock = RLock()
_trace_api_loading = False
_call_tracer: Any = None
_call_tracer_loading = False
_NOOP_CONTEXT = nullcontext()
_run_span_clients: weakref.WeakValueDictionary[int, _SolwynBase] = weakref.WeakValueDictionary()


def _reset_after_fork() -> None:
    global _trace_api_lock, _trace_api_loading, _call_tracer, _call_tracer_loading
    _trace_api_lock = RLock()
    _trace_api_loading = False
    _call_tracer = None
    _call_tracer_loading = False


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


def _get_call_tracer(trace: ModuleType) -> Any:
    """Cache even a proxy tracer; guard provider callbacks against re-entry."""
    global _call_tracer, _call_tracer_loading
    if _call_tracer is None:
        with _trace_api_lock:
            if _call_tracer is None:
                if _call_tracer_loading:
                    return None
                _call_tracer_loading = True
                try:
                    from solwyn import __version__

                    _call_tracer = trace.get_tracer(
                        "solwyn", instrumenting_library_version=__version__
                    )
                finally:
                    _call_tracer_loading = False
    return _call_tracer


def _register_run_span_client(client: _SolwynBase) -> None:
    """Let client-independent run scopes reuse a live client's existing opt-in."""
    if client._solwyn_config.otel_spans:
        with _trace_api_lock:
            _run_span_clients[id(client)] = client


def _run_spans_enabled() -> bool:
    if os.environ.get("SOLWYN_OTEL_SPANS", "").lower() in ("true", "1", "yes"):
        return True
    if not _run_span_clients:
        return False
    with _trace_api_lock:
        return any(client._solwyn_config.otel_spans for client in _run_span_clients.values())


def _span_is_exportable(span: Any) -> bool:
    """Never adopt or propagate an allocated but unexported child identifier."""
    context = span.get_span_context()
    return bool(span.is_recording() and context.is_valid and context.trace_flags.sampled)


class _RunSpanActivation:
    """One context-local attachment, independent of the logical run's lifetime."""

    def __init__(self, api: ModuleType, token: Any, prior: Any) -> None:
        self._api = api
        self._token = token
        self._prior = prior

    def detach(self, *, restore: bool = False) -> None:
        # A copied-context finalizer cannot reset the original token. Restore
        # its prior context just like the run ContextVar fallback. Also handle
        # custom OTel runtimes whose detach raises rather than swallowing it.
        if not restore:
            try:
                self._api.detach(self._token)
                return
            except BaseException:
                pass
        with suppress(BaseException):
            self._api.attach(self._prior)


class _RunSpan:
    """One run span shared by all activations, ended once without exception data."""

    def __init__(self, span: Any = None, trace: ModuleType | None = None) -> None:
        self._span = span
        self._trace = trace
        self._lock = Lock()
        self._ended = False
        self._error = False

    def attach(self) -> _RunSpanActivation | None:
        if self._span is None or self._trace is None:
            return None
        api = None
        prior = None
        token = None
        transferred = False
        try:
            from opentelemetry import context

            api = context
            prior = context.get_current()
            token = context.attach(self._trace.set_span_in_context(self._span))
            activation = _RunSpanActivation(context, token, prior)
            transferred = True
            return activation
        except Exception:
            return None
        finally:
            # An interrupt can land after attach mutates the context, before
            # its token/owner reaches the scope. Retire any token we obtained
            # and restore the saved context even when attach never returned.
            if not transferred and api is not None:
                if token is not None:
                    with suppress(BaseException):
                        api.detach(token)
                if prior is not None:
                    with suppress(BaseException):
                        api.attach(prior)

    def failed(self) -> None:
        if self._span is not None:
            with self._lock:
                self._error = True

    def finalize_on_gc(self, owner: object) -> weakref.finalize[..., object] | None:
        if self._span is None:
            return None
        finalizer = weakref.finalize(owner, self.end, error=True)
        finalizer.atexit = False
        return finalizer

    def end(self, *, error: bool = False) -> None:
        if self._span is None or self._trace is None:
            return
        with self._lock:
            if self._ended:
                return
            self._ended = True
            error |= self._error
        try:
            self._span.set_status(
                self._trace.StatusCode.ERROR if error else self._trace.StatusCode.OK
            )
        except Exception:
            pass
        finally:
            with suppress(Exception):
                self._span.end()


_INACTIVE_RUN_SPAN = _RunSpan()


def _start_run_span(run_id: str, parent_run_id: str | None) -> _RunSpan:
    """Create a detached run span only when opted in and eligible for export."""
    if not _run_spans_enabled():
        return _INACTIVE_RUN_SPAN
    span = None
    trace = None
    transferred = False
    try:
        trace = _get_trace_api()
        if trace is None:
            return _INACTIVE_RUN_SPAN
        span = trace.get_tracer("solwyn").start_span("solwyn.run", kind=trace.SpanKind.INTERNAL)
        if _span_is_exportable(span):
            span.set_attribute("solwyn.agent_run_id", run_id)
            if parent_run_id is not None:
                span.set_attribute("solwyn.parent_agent_run_id", parent_run_id)
            owner = _RunSpan(span, trace)
            transferred = True
            return owner
    except BaseException as exc:
        if span is not None and trace is not None:
            with suppress(Exception):
                span.set_status(trace.StatusCode.ERROR)
        if not isinstance(exc, Exception):
            raise
    finally:
        if span is not None and not transferred:
            with suppress(Exception):
                span.end()
    return _INACTIVE_RUN_SPAN


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


def capture_trace(span: Any = None) -> _TraceSnapshot | None:
    """Snapshot a supplied or current valid span without attaching a context."""
    try:
        trace = _get_trace_api()
        if trace is None:
            return None
        context = (span if span is not None else trace.get_current_span()).get_span_context()
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
        self.active = span is not None
        self._span = span
        self._trace = trace
        self._lock = Lock()
        self._ended = False
        self._deferred = False
        self._failover_reason: str | None = None
        self._end_holds = 0
        self._end_requested = False
        self._error = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, typ: object, exc: BaseException | None, tb: object) -> None:
        if exc is not None or not self._deferred:
            self.end(error=exc is not None, exception=exc)

    @contextmanager
    def current(self) -> Iterator[None]:
        """Attach only for provider dispatch; never across a stream return."""
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
        return capture_trace(self._span)

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

    @contextmanager
    def finishing(self) -> Iterator[None]:
        """Keep completion open through nested provider close/exit cleanup.

        Completion can request end before the provider finishes closing. Retain
        only an error flag until the outer cleanup returns; never retain
        exceptions or make the span current during application cleanup.
        """
        if self._span is None:
            yield
            return
        with self._lock:
            self._end_holds += 1
        try:
            yield
        except BaseException as exc:
            self.end(error=True, exception=exc)
            raise
        finally:
            with self._lock:
                self._end_holds -= 1
                end_requested = self._end_requested and self._end_holds == 0
            if end_requested:
                self.end()

    def finalize_on_gc(self, owner: object) -> weakref.finalize[..., object] | None:
        if self._span is None:
            return None
        finalizer = weakref.finalize(owner, self.end, error=True)
        finalizer.atexit = False
        return finalizer

    def end(self, *, error: bool = False, exception: BaseException | None = None) -> None:
        if self._span is None or self._trace is None:
            return
        with self._lock:
            if self._ended:
                return
            self._error |= error or isinstance(exception, RunStoppedError)
            self._end_requested = True
            if self._end_holds:
                return
            self._ended = True
        try:
            if self._failover_reason is not None:
                self._span.set_attribute("solwyn.failover.reason", self._failover_reason)
            self._span.set_status(
                self._trace.Status(
                    self._trace.StatusCode.ERROR if self._error else self._trace.StatusCode.OK
                )
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
    trace = None
    transferred = False
    try:
        trace = _get_trace_api()
        if trace is None:
            return _INACTIVE_CALL_SPAN
        tracer = _get_call_tracer(trace)
        if tracer is None:
            return _INACTIVE_CALL_SPAN
        span = tracer.start_span("solwyn.call", kind=trace.SpanKind.INTERNAL)
        if _span_is_exportable(span):
            owner = _CallSpan(span, trace)
            transferred = True
            return owner
    except BaseException as exc:
        if span is not None and trace is not None:
            with suppress(Exception):
                span.set_status(trace.Status(trace.StatusCode.ERROR))
        if not isinstance(exc, Exception):
            raise
    finally:
        # An interrupt between start_span and ownership transfer must not strand
        # a live span, even when a custom span's context/recording query raises.
        if span is not None and not transferred:
            with suppress(Exception):
                span.end()
    return _INACTIVE_CALL_SPAN
