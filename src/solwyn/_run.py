"""Agent-run scopes for context managers and begin/end-shaped adapters.

Binds an active run id/name plus optional explicit customer tags to a
``ContextVar`` for the duration of a scope. Cost events emitted inside the
scope carry a copied snapshot in the wire payload; events emitted outside are
unscoped. Tags are never inferred from prompts or responses.

The contextvar is the scope integration seam. Each intercepted call captures
an immutable-by-convention snapshot and threads it through deferred reporting.
``contextvars`` propagation guarantees correct scoping across asyncio tasks.
Sync generators stepped in fresh contexts resolve their scope from their
executing frame, without retaining the frame or binding state to a worker.
Threads and ``ThreadPoolExecutor`` workers do not
inherit the active run reliably; use ``run_in_executor(...)`` or
``contextvars.copy_context().run(...)`` when submitting threaded work.
A run recovered only from a generator frame is not present in copied contexts;
activate a detached handle explicitly in threads started inside that step.
Do not open a run scope inside an async generator; async generator yields
share the consumer's context and would leak the generator's run into the
consumer body.

``start_run(...)`` exposes the same scope machinery to framework adapters
whose boundaries arrive as separate callbacks. Its ``RunHandle.finish()``
must run in the same context that called ``start_run(...)``.

``create_run(...)`` instead snapshots a detached logical identity without
changing the current context. Its handle can bind that identity repeatedly in
different tasks or threads through short-lived ``activate()`` scopes.

This module never touches prompt or response content.
"""

from __future__ import annotations

import inspect
import os
import sys
import unicodedata
import uuid
import warnings
import weakref
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future
from contextlib import AbstractAsyncContextManager, AbstractContextManager, suppress
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass, field
from threading import Lock
from types import CodeType, FrameType, TracebackType
from typing import NamedTuple, ParamSpec, TypeVar

from solwyn import _otel
from solwyn._constants import (
    AGENT_RUN_NAME_MAX_LENGTH,
    TAG_KEY_MAX_LENGTH,
    TAG_VALUE_MAX_LENGTH,
    TAGS_MAX_KEYS,
)
from solwyn._lifecycle import _surrender_run, register_fork_reset
from solwyn.exceptions import SolwynTagsClampedWarning

_P = ParamSpec("_P")
_T = TypeVar("_T")
_RunContextSnapshot = tuple[
    str | None,
    str | None,
    dict[str, str] | None,
    str | None,
]


class RunContext(NamedTuple):
    """Public snapshot of the active agent-run scope."""

    id: str | None
    name: str | None
    tags: dict[str, str] | None


@dataclass(frozen=True)
class _RunIdentity:
    """Immutable logical identity reused by detached run activations."""

    run_id: str
    name: str
    tags: tuple[tuple[str, str], ...] | None
    parent_run_id: str | None

    def active_value(self) -> tuple[str, str, dict[str, str] | None]:
        """Return a fresh ContextVar payload for one activation."""
        tags = dict(self.tags) if self.tags is not None else None
        return (self.run_id, self.name, tags)


# Single contextvar holding either None or one run payload. Storing all values
# together makes the swap atomic across async task transitions.
_active_run: ContextVar[tuple[str, str, dict[str, str] | None] | None] = ContextVar(
    "solwyn_active_run", default=None
)

_DISALLOWED_NAME_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_OwnerKey = tuple[int, CodeType] | None


@dataclass(eq=False)
class _OpenRun:
    """Scope-owned lifetime, independent of any ContextVar token or frame."""

    identity: _RunIdentity
    active: tuple[str, str, dict[str, str] | None]
    span: _otel._RunSpan
    owner_key: _OwnerKey
    closed: bool = False
    lock: Lock = field(default_factory=Lock)

    def _reset_after_fork_in_child(self) -> None:
        self.lock = Lock()

    @property
    def run_id(self) -> str:
        return self.identity.run_id

    @property
    def parent_run_id(self) -> str | None:
        return self.identity.parent_run_id

    @property
    def generator_owned(self) -> bool:
        return self.owner_key is not None and bool(
            self.owner_key[1].co_flags & inspect.CO_GENERATOR
        )


@dataclass(frozen=True)
class _RunFrame:
    scope_id: int
    record: _OpenRun
    token: Token[tuple[str, str, dict[str, str] | None] | None]
    prior_active: tuple[str, str, dict[str, str] | None] | None
    span_activation: _otel._RunSpanActivation | None

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def span(self) -> _otel._RunSpan:
        return self.record.span


_run_frames: ContextVar[tuple[_RunFrame, ...]] = ContextVar("solwyn_run_frames", default=())

# No frame is retained here or on a record: a suspended generator must remain
# collectible. Frame ids are stable across next() on supported CPython versions;
# matching the code too guards against stale entries after address reuse.
_generator_runs: dict[int, list[tuple[CodeType, weakref.ReferenceType[_OpenRun]]]] = {}
_generator_runs_lock = Lock()


def _reset_after_fork() -> None:
    global _generator_runs_lock
    _generator_runs_lock = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _owner_key() -> _OwnerKey:
    frame = None
    try:
        frame = sys._getframe(2)
        while frame is not None and frame.f_globals.get("__name__") == __name__:
            frame = frame.f_back
        return (id(frame), frame.f_code) if frame is not None else None
    except (AttributeError, ValueError):
        return None
    finally:
        del frame


def _unregister_generator(key: _OwnerKey, reference: weakref.ReferenceType[_OpenRun]) -> None:
    if key is None:
        return
    with _generator_runs_lock:
        entries = _generator_runs.get(key[0], [])
        entries = [(code, ref) for code, ref in entries if ref is not reference]
        if entries:
            _generator_runs[key[0]] = entries
        else:
            _generator_runs.pop(key[0], None)


def _register_generator(record: _OpenRun) -> None:
    key = record.owner_key
    if key is None or not key[1].co_flags & inspect.CO_GENERATOR:
        return
    reference = weakref.ref(record, lambda ref: _unregister_generator(key, ref))
    with _generator_runs_lock:
        _generator_runs.setdefault(key[0], []).append((key[1], reference))


def _close_record(record: _OpenRun) -> bool:
    """Claim the open-to-closed transition once, before performing cleanup."""
    with record.lock:
        if record.closed:
            return False
        record.closed = True
    key = record.owner_key
    if key is not None and key[1].co_flags & inspect.CO_GENERATOR:
        with _generator_runs_lock:
            entries = tuple(_generator_runs.get(key[0], ()))
        for _, reference in entries:
            if reference() is record:
                _unregister_generator(key, reference)
    return True


def _resolve_active() -> _OpenRun | None:
    """Resolve ContextVar scoping plus runs on the executing generator stack.

    A binding opened in this step (same owner frame or closer) shadows a
    generator fallback. An inherited driver binding cannot shadow it. Nothing
    is attached to the worker thread or copied into the driver's context.
    """
    frames = _run_frames.get()
    # Ordinary scopes allow copied task contexts to outlive the scope and
    # acquire a new lease. Preserve that existing public contract. Generator
    # records retire globally so old step contexts cannot resurrect a closed run.
    current = next(
        (
            item.record
            for item in reversed(frames)
            if not item.record.closed or not item.record.generator_owned
        ),
        None,
    )
    if not _generator_runs:
        return current
    visible = {item.record for item in frames}
    frame = None
    try:
        frame = sys._getframe(1)
        for _ in range(256):
            if frame is None:
                break
            key = (id(frame), frame.f_code)
            if (
                current is not None
                and current.owner_key is not None
                and current.owner_key[0] == key[0]
                and current.owner_key[1] is key[1]
            ):
                return current
            with _generator_runs_lock:
                entries = tuple(_generator_runs.get(key[0], ()))
            # Dereference outside the lock: dropping the last temporary strong
            # reference may invoke the weakref callback, which also takes it.
            for code, reference in reversed(entries):
                record = reference()
                if (
                    code is key[1]
                    and record is not None
                    and not record.closed
                    and record not in visible
                ):
                    return record
            frame = frame.f_back
    except (AttributeError, ValueError):
        pass
    finally:
        del frame
    return current


def _fallback_run_span() -> _otel._RunSpan | None:
    record = _resolve_active()
    if record is not None and not any(item.record is record for item in _run_frames.get()):
        return record.span
    return None


def _pop_frame(frame: _RunFrame, frames: tuple[_RunFrame, ...]) -> None:
    restore = False
    try:
        _active_run.reset(frame.token)
    except (ValueError, RuntimeError):
        _active_run.set(frame.prior_active)
        restore = True
    _run_frames.set(frames[:-1])
    if frame.span_activation is not None:
        frame.span_activation.detach(restore=restore)


def _new_run_id() -> str:
    """Generate a fresh agent_run_id.

    Format: ``run_`` + UUID4 canonical string.
    Comfortably under the 255-char wire cap.
    """
    return f"run_{uuid.uuid4()}"


def current_run() -> tuple[str | None, str | None]:
    """Return the active ``(agent_run_id, agent_run_name)`` or ``(None, None)``.

    Read at metadata-event build time. Returning a tuple (rather than the
    raw ``ContextVar`` value) lets callers unpack without a None-check.
    """
    record = _resolve_active()
    if record is None:
        return (None, None)
    active = record.active
    return (active[0], active[1])


def current_run_context() -> RunContext:
    """Return the active run id, name, and a defensive copy of its tags."""
    record = _resolve_active()
    if record is None:
        return RunContext(None, None, None)
    active = record.active
    tags = dict(active[2]) if active[2] is not None else None
    return RunContext(active[0], active[1], tags)


def _copy_tags(
    tags: object | None,
    *,
    parameter: str,
) -> dict[str, str] | None:
    """Validate and copy one explicit tag mapping without normalization."""
    if tags is None:
        return None
    if not isinstance(tags, Mapping):
        raise TypeError(f"{parameter} requires a mapping of string keys to string values")
    if len(tags) > TAGS_MAX_KEYS:
        raise ValueError(f"{parameter} allows at most {TAGS_MAX_KEYS} keys")

    copied: dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str):
            raise TypeError(f"{parameter} keys must be strings")
        if not key:
            raise ValueError(f"{parameter} keys must be non-empty")
        if len(key) > TAG_KEY_MAX_LENGTH:
            raise ValueError(f"{parameter} key exceeds max length {TAG_KEY_MAX_LENGTH}")
        if "\x00" in key:
            raise ValueError(f"{parameter} keys must not contain NUL characters")
        if not isinstance(value, str):
            raise TypeError(f"{parameter} values must be strings")
        if len(value) > TAG_VALUE_MAX_LENGTH:
            raise ValueError(f"{parameter} value exceeds max length {TAG_VALUE_MAX_LENGTH}")
        if "\x00" in value:
            raise ValueError(f"{parameter} values must not contain NUL characters")
        copied[key] = value
    return copied or None


def _capture_run_context(
    per_call_tags: object | None = None,
    *,
    default_tags: object | None = None,
) -> _RunContextSnapshot:
    """Capture the active run, immediate parent, and merged tag layers."""
    record = _resolve_active()
    run_id: str | None = None
    run_name: str | None = None
    parent_run_id: str | None = None
    defaults = _copy_tags(default_tags, parameter="client tags")
    scope_tags: dict[str, str] | None = None
    if record is not None:
        run_id, run_name, scope_tags = record.active
        parent_run_id = record.parent_run_id
    override = _copy_tags(per_call_tags, parameter="solwyn_tags")

    merged: dict[str, str] = {}
    clamped = False
    for layer in (override, scope_tags, defaults):
        if layer is None:
            continue
        for key, value in layer.items():
            if key in merged:
                continue
            if len(merged) == TAGS_MAX_KEYS:
                clamped = True
                continue
            merged[key] = value

    if clamped:
        with suppress(Exception):
            warnings.warn(
                f"merged tags exceed {TAGS_MAX_KEYS} keys; lower-priority tags were dropped",
                SolwynTagsClampedWarning,
                stacklevel=2,
            )
    return (run_id, run_name, merged or None, parent_run_id)


def _name_has_disallowed_char(name: str) -> bool:
    return any(unicodedata.category(char) in _DISALLOWED_NAME_CATEGORIES for char in name)


def _validate_run_definition(
    name: object,
    tags: object | None,
    *,
    api_name: str,
) -> tuple[str, dict[str, str] | None]:
    """Validate one run name/tag definition without reading active context."""
    if not isinstance(name, str):
        raise TypeError(f"{api_name}(name) requires str, got {type(name).__name__}")
    if not name.strip():
        raise ValueError(f"{api_name}(name) requires a non-empty name")
    if _name_has_disallowed_char(name):
        raise ValueError(
            f"{api_name}(name) must not contain control characters, "
            "format, or line-separator characters"
        )
    if len(name) > AGENT_RUN_NAME_MAX_LENGTH:
        raise ValueError(
            f"{api_name}(name) exceeds max length {AGENT_RUN_NAME_MAX_LENGTH} (got {len(name)})"
        )
    copied_tags = _copy_tags(tags, parameter=f"{api_name}(tags)")
    return name, copied_tags


def _snapshot_run_identity(
    name: str,
    tags: Mapping[str, str] | None,
    *,
    inherit_tags: bool,
) -> _RunIdentity:
    """Create one logical identity from the current parent and tag context."""
    prior = _resolve_active()
    prior_active = prior.active if prior is not None else None
    parent_run_id = prior.run_id if prior is not None else None
    scope_tags = dict(tags or {})
    if inherit_tags and prior_active is not None and prior_active[2] is not None:
        for key, value in prior_active[2].items():
            scope_tags.setdefault(key, value)
    return _RunIdentity(
        run_id=_new_run_id(),
        name=name,
        tags=tuple(scope_tags.items()) or None,
        parent_run_id=parent_run_id,
    )


def _called_from_async_generator() -> bool:
    """Return True when ``run()`` is being entered inside an async generator."""
    frame: FrameType | None = None
    try:
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code.co_flags & inspect.CO_ASYNC_GENERATOR:
                return True
            frame = frame.f_back
        return False
    except (AttributeError, ValueError):
        return False
    finally:
        del frame


class _RunScope(AbstractContextManager[str], AbstractAsyncContextManager[str]):
    """Context manager returned by ``solwyn.run(name)``.

    Supports both ``with`` and ``async with``. Nested scopes inherit tags by
    default, with inner values winning only on conflicting keys.
    The outer scope is restored automatically on exit via ``ContextVar.reset``.
    """

    def __init__(
        self,
        name: str,
        tags: Mapping[str, str] | None = None,
        *,
        inherit_tags: bool = True,
    ) -> None:
        self._name, self._tags = _validate_run_definition(
            name,
            tags,
            api_name="solwyn.run",
        )
        self._inherit_tags = inherit_tags
        self._scope_id = id(self)
        self._open_runs: dict[_OwnerKey, list[_OpenRun]] = {}
        self._records_lock = Lock()
        register_fork_reset(self)

    def _reset_after_fork_in_child(self) -> None:
        self._records_lock = Lock()

    def _enter(self) -> str:
        if _called_from_async_generator():
            raise TypeError(
                "solwyn.run(...) inside async generators is not supported; "
                "open the scope in the consumer or await the generator inside an outer scope"
            )
        prior_active = _active_run.get()
        frames = _run_frames.get()
        identity = _snapshot_run_identity(
            self._name,
            self._tags,
            inherit_tags=self._inherit_tags,
        )
        span = _otel._start_run_span(
            identity.run_id, identity.parent_run_id, parent=_fallback_run_span()
        )
        record = None
        activation = None
        token = None
        try:
            record = _OpenRun(identity, identity.active_value(), span, _owner_key())
            register_fork_reset(record)
            with self._records_lock:
                self._open_runs.setdefault(record.owner_key, []).append(record)
            _register_generator(record)
            activation = span.attach()
            token = _active_run.set(record.active)
            _run_frames.set(
                (
                    *frames,
                    _RunFrame(
                        scope_id=self._scope_id,
                        record=record,
                        token=token,
                        prior_active=prior_active,
                        span_activation=activation,
                    ),
                )
            )
        except BaseException:
            with suppress(BaseException):
                if activation is not None:
                    activation.detach()
                if token is not None:
                    _active_run.reset(token)
            if record is not None:
                self._close(record, error=True)
            else:
                with suppress(BaseException):
                    span.end(error=True)
            raise
        return identity.run_id

    def _find_record(self, frames: tuple[_RunFrame, ...]) -> _OpenRun | None:
        key = _owner_key()
        with self._records_lock:
            owned = self._open_runs.get(key) if key is not None else None
            visible = next(
                (frame for frame in reversed(frames) if frame.scope_id == self._scope_id), None
            )
            # A closed record can still have a token in its original context.
            # An executing generator identifies its live invocation. Explicit
            # later cleanup pops its closed frame before an outer invocation
            # of this same scope object.
            if owned and (
                visible is None
                or not visible.record.closed
                or (key is not None and key[1].co_flags & inspect.CO_GENERATOR)
            ):
                return owned[-1]
            if visible is not None:
                return visible.record
            # Explicit cleanup from a different caller is unambiguous only
            # when this scope owns one open record. Never guess by insertion order.
            records = [record for stack in self._open_runs.values() for record in stack]
            return records[0] if len(records) == 1 else None

    def _forget(self, record: _OpenRun) -> None:
        with self._records_lock:
            stack = self._open_runs.get(record.owner_key, [])
            stack = [item for item in stack if item is not record]
            if stack:
                self._open_runs[record.owner_key] = stack
            else:
                self._open_runs.pop(record.owner_key, None)

    def _close(self, record: _OpenRun, *, error: bool) -> None:
        if not _close_record(record):
            return
        try:
            with suppress(BaseException):
                _surrender_run(record.run_id)
        finally:
            self._forget(record)
            with suppress(BaseException):
                record.span.end(error=error)

    def _exit(self, *, require_same_context: bool = False, error: bool = False) -> None:
        """Retire our own record once; only pop tokens visible in this context."""
        frames = _run_frames.get()
        record = self._find_record(frames)
        matching = next((item for item in reversed(frames) if item.record is record), None)
        if matching is None:
            if require_same_context:
                raise RuntimeError(
                    "RunHandle.finish() must be called in the same context where "
                    "start_run() created it"
                )
            if record is not None:
                self._close(record, error=error)
            return
        if frames[-1] is not matching:
            raise RuntimeError("solwyn.run scopes must exit in LIFO order")
        if require_same_context:
            try:
                _active_run.reset(matching.token)
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    "RunHandle.finish() must be called in the same context where "
                    "start_run() created it"
                ) from exc
            _run_frames.set(frames[:-1])
            if matching.span_activation is not None:
                matching.span_activation.detach()
            # The handle owns surrender and span end, after releasing its lock.
            _close_record(matching.record)
            self._forget(matching.record)
            return
        try:
            _pop_frame(matching, frames)
        finally:
            self._close(matching.record, error=error)

    def __enter__(self) -> str:
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit(error=exc_type is not None)

    async def __aenter__(self) -> str:
        return self._enter()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit(error=exc_type is not None)


class RunHandle:
    """Lifecycle handle for framework adapter run identities.

    A handle from :func:`start_run` owns one ContextVar-backed scope and must
    finish in its creating context. A handle from :func:`create_run` owns a
    detached identity that can be bound through reusable :meth:`activate`
    scopes in different contexts before it is finished.
    """

    def __init__(
        self,
        scope: _RunScope | None,
        run_id: str,
        *,
        identity: _RunIdentity | None = None,
        span: _otel._RunSpan | None = None,
    ) -> None:
        if (scope is None) == (identity is None):
            raise RuntimeError("run handle requires exactly one scope or detached identity")
        self._scope = scope
        self._run_id = run_id
        self._identity = identity
        self._finished = False
        self._active_activations = 0
        self._state_lock = Lock()
        if span is None:
            span = _otel._start_run_span(
                run_id,
                identity.parent_run_id if identity is not None else None,
                parent=_fallback_run_span(),
            )
        self._span = span
        self._span_finalizer = self._span.finalize_on_gc(self)

    @property
    def run_id(self) -> str:
        """Return the stable id for this run scope."""
        return self._run_id

    def finish(self) -> None:
        """Finish this logical run, restoring a started scope when applicable.

        Finishing ends the run for both handle flavours, so both release the
        leases it held — outside the handle lock, and only once the finish has
        actually taken (a refused finish leaves the run running). An
        ``activate()`` scope deliberately does NOT: a detached identity is
        re-entered by design, and releasing at each activation would trade one
        blocking grant per binding for nothing.
        """
        with self._state_lock:
            if self._finished:
                raise RuntimeError(f"run handle {self._run_id!r} already finished")
            if self._active_activations:
                raise RuntimeError(
                    f"run handle {self._run_id!r} cannot finish while activations are still active"
                )
            if self._scope is not None:
                self._scope._exit(require_same_context=True)
            self._finished = True
        try:
            _surrender_run(self._run_id)
        finally:
            self._span.end()
            if self._span_finalizer is not None:
                self._span_finalizer.detach()

    def activate(self) -> _RunActivation:
        """Bind a detached logical identity in the caller's context."""
        if self._identity is None:
            raise RuntimeError("RunHandle.activate() requires a handle created by create_run()")
        return _RunActivation(self)

    def _reserve_activation(self) -> _RunIdentity:
        with self._state_lock:
            if self._finished:
                raise RuntimeError(f"run handle {self._run_id!r} already finished")
            if self._identity is None:
                raise RuntimeError("RunHandle.activate() requires a handle created by create_run()")
            self._active_activations += 1
            return self._identity

    def _release_activation(self) -> None:
        with self._state_lock:
            if self._active_activations <= 0:
                raise RuntimeError("run handle activation count is inconsistent")
            self._active_activations -= 1


class _RunActivation(AbstractContextManager[str], AbstractAsyncContextManager[str]):
    """One strict ContextVar binding of a detached logical run identity."""

    def __init__(self, handle: RunHandle) -> None:
        self._handle = handle
        self._scope_id = id(self)
        self._entered = False
        self._record: _OpenRun | None = None

    def _enter(self) -> str:
        frames = _run_frames.get()
        current = _resolve_active()
        if (current is not None and current.run_id == self._handle.run_id) or any(
            frame.run_id == self._handle.run_id and not frame.record.closed for frame in frames
        ):
            raise RuntimeError(
                f"run handle {self._handle.run_id!r} is already active in this context"
            )
        identity = self._handle._reserve_activation()
        record = None
        prior_active = _active_run.get()
        token: Token[tuple[str, str, dict[str, str] | None] | None] | None = None
        activation = None
        try:
            record = _OpenRun(identity, identity.active_value(), self._handle._span, _owner_key())
            self._record = record
            register_fork_reset(record)
            _register_generator(record)
            activation = self._handle._span.attach()
            token = _active_run.set(record.active)
            _run_frames.set(
                (
                    *frames,
                    _RunFrame(
                        scope_id=self._scope_id,
                        record=record,
                        token=token,
                        prior_active=prior_active,
                        span_activation=activation,
                    ),
                )
            )
        except BaseException:
            with suppress(BaseException):
                if activation is not None:
                    activation.detach()
                if token is not None:
                    _active_run.reset(token)
            if record is not None:
                self._deactivate(record, error=False)
            else:
                self._handle._release_activation()
            raise
        self._entered = True
        return identity.run_id

    def _exit(self, *, error: bool = False) -> None:
        frames = _run_frames.get()
        record = self._record
        key = _owner_key()
        owned = (
            record is not None and not record.closed and key is not None and record.owner_key == key
        )
        for frame in reversed(frames):
            if frame.scope_id == self._scope_id and (not owned or frame.record.owner_key == key):
                record = frame.record
                break
        matching_index = next(
            (index for index, frame in enumerate(frames) if frame.record is record),
            None,
        )
        if matching_index is None:
            if record is not None:
                self._deactivate(record, error=error)
            return
        if matching_index != len(frames) - 1:
            raise RuntimeError("solwyn.run scopes must exit in LIFO order")
        try:
            _pop_frame(frames[-1], frames)
        finally:
            self._deactivate(frames[-1].record, error=error)

    def _deactivate(self, record: _OpenRun, *, error: bool) -> None:
        if _close_record(record):
            if error:
                record.span.failed()
            self._handle._release_activation()
            self._entered = False

    def __enter__(self) -> str:
        if self._entered:
            raise RuntimeError("RunHandle activation is already entered")
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit(error=exc_type is not None)

    async def __aenter__(self) -> str:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._exit(error=exc_type is not None)


def start_run(
    name: str,
    tags: Mapping[str, str] | None = None,
    *,
    inherit_tags: bool = True,
) -> RunHandle:
    """Open a run scope for begin/end-shaped framework callbacks."""
    scope = _RunScope(name, tags, inherit_tags=inherit_tags)
    run_id = scope._enter()
    return RunHandle(scope, run_id, span=_run_frames.get()[-1].span)


def create_run(
    name: str,
    tags: Mapping[str, str] | None = None,
    *,
    inherit_tags: bool = True,
) -> RunHandle:
    """Create a detached logical run without changing the current context.

    The returned handle snapshots its id, name, inherited tags, and parent at
    creation. Use ``with handle.activate():`` around work in any task or thread;
    each activation binds that same logical identity with a fresh ContextVar
    token. Call :meth:`RunHandle.finish` after every activation has exited.
    """
    validated_name, copied_tags = _validate_run_definition(
        name,
        tags,
        api_name="solwyn.create_run",
    )
    identity = _snapshot_run_identity(
        validated_name,
        copied_tags,
        inherit_tags=inherit_tags,
    )
    return RunHandle(None, identity.run_id, identity=identity)


def run_in_executor(
    executor: Executor,
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> Future[_T]:
    """Submit ``fn`` to an executor with the active ``solwyn.run(...)`` tag preserved.

    ``ThreadPoolExecutor`` does not copy ``ContextVar`` values into worker
    threads. This helper wraps ``fn`` in ``copy_context().run`` so the
    active run id propagates to the worker.

    Returns:
        The executor's ``concurrent.futures.Future[T]``, not an awaitable.
        Call ``.result()`` to wait for the value. In asyncio code, bridge it
        with ``asyncio.wrap_future(future)``.

    Args:
        executor: Any ``concurrent.futures.Executor``.
        fn: Callable to run in a worker.
        *args, **kwargs: Forwarded to ``fn`` through the copied context.

    Note:
        If ``executor.shutdown(cancel_futures=True)`` cancels work before it
        starts, ``future.result()`` raises ``CancelledError``.
    """
    ctx = copy_context()

    def run_with_context() -> _T:
        return ctx.run(fn, *args, **kwargs)

    return executor.submit(run_with_context)


def run(
    name: str,
    tags: Mapping[str, str] | None = None,
    *,
    inherit_tags: bool = True,
) -> _RunScope:
    """Open an agent-run scope.

    Cost events emitted inside the scope are tagged with a fresh stable
    ``agent_run_id``, the provided ``name``, and a copied snapshot of optional
    explicit customer ``tags``. Use as a sync or async context manager::

        with solwyn.run("nightly-batch", tags={"team": "research"}) as run_id:
            client.chat.completions.create(...)

        async with solwyn.run("ingest-job") as run_id:
            await aclient.chat.completions.create(...)

    Nested runs inherit outer tags additively by default. Inner values win
    only where they reuse an outer key. Pass ``inherit_tags=False`` to start
    a fresh tag scope. The inner run always gets its own id, and the outer
    context is restored after the inner ``__exit__``. Sequential scopes get
    distinct ids — the API aggregates by id, not name.

    Tasks created with ``asyncio.create_task(...)`` inside an ordinary scope
    capture that task-local context, including after the scope exits. Generator
    records instead retire on exit across all copied step contexts. Use
    ``asyncio.TaskGroup`` or await spawned tasks before exiting when attribution
    must be bounded.

    ``solwyn.run(...)`` is not supported inside async generators. A scope
    opened before ``yield`` would remain active in the consumer's ``async for``
    body, so the SDK raises at scope entry instead.

    When an operator stops this run from the Solwyn dashboard, per-call traffic
    raises :class:`solwyn.RunStoppedError` on its next budget check. Leased
    traffic raises only after a lease renewal or re-grant learns the stop.
    Requests already in flight and streams already returned are not interrupted;
    control-plane connectivity failures retain the configured fail-open posture.
    """
    return _RunScope(name, tags, inherit_tags=inherit_tags)
